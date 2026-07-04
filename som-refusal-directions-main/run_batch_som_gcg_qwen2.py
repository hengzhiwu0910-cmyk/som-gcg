import os
import gc
import json
import torch

from models.load_models import load_model
from transformers import AutoTokenizer
from nanogcg import GCGConfig
import nanogcg
from models.system_prompts import qwen_sys   # 只使用 qwen_sys


def read_json_or_jsonl(path):
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path, item):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_existing_keys(path):
    """
    用于断点续跑。只跳过已经成功生成 best_suffix 的 prompt。
    error 记录不算完成。
    """
    done = set()
    if not os.path.exists(path):
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                # 只把成功记录加入 done
                if "prompt" in item and "best_suffix" in item and "error" not in item:
                    done.add(item["prompt"])
            except Exception:
                pass

    return done


def get_prompt(item):
    """
    兼容不同字段名。
    target 文件里如果字段名就是 prompt，会直接用 prompt。
    """
    for key in ["prompt", "behavior", "harmful_prompt", "instruction"]:
        if key in item and item[key]:
            return item[key]
    raise KeyError(f"Cannot find prompt field in item keys: {list(item.keys())}")


def get_target(item, target_key="target_64"):
    if target_key in item and item[target_key]:
        return item[target_key]
    raise KeyError(
        f"Cannot find {target_key}. Available keys: {list(item.keys())}"
    )


def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # =========================
    # 主要配置
    # =========================
    input_path = "targets/qwen2_som_ablation_targets.json"
    output_path = "results/qwen2_only_gcg_suffixes_val_alllayer_target64.jsonl"

    target_key = "target_64"
    start_idx = 0
    max_items = 41

    model_name = "qwen2-7b"
    device = "cuda"

    print(f"[*] nanogcg path: {nanogcg.__file__}")
    print(f"[*] Loading target model: {model_name}")

    # 加载模型（使用 load_model，它会返回带有 tokenizer 和 model 的对象）
    model_obj = load_model(model_name, device=device)

    if hasattr(model_obj, "model"):
        raw_model = model_obj.model
    else:
        raw_model = model_obj

    raw_model.eval()

    if hasattr(model_obj, "tokenizer"):
        tokenizer = model_obj.tokenizer
    else:
        hf_path = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============================================================
    # GCG 配置：关闭 filter_ids 避免解码重编码不一致报错
    # ============================================================
    config = GCGConfig(
        num_steps=300,
        search_width=512,
        topk=512,
        optim_str_init="x x x x x x x x x x x x x x x x x x x x",
        early_stop=False,
        buffer_size=8,
        verbosity="INFO",
        use_prefix_cache=True,
        use_resid_write_hooks=False,
        som_loss_scale=1.0,
        enable_som_loss=False,
        filter_ids=False,          # 关键：跳过 token 过滤
    )

    data = read_json_or_jsonl(input_path)

    if max_items is not None:
        selected_data = data[start_idx:start_idx + max_items]
    else:
        selected_data = data[start_idx:]

    done_prompts = load_existing_keys(output_path)

    print(f"[*] Loaded {len(data)} total items.")
    print(f"[*] Running items from index {start_idx}, count={len(selected_data)}.")
    print(f"[*] Already completed in this output file: {len(done_prompts)}")
    print(f"[*] Saving to: {output_path}")

    for local_i, item in enumerate(selected_data):
        global_i = start_idx + local_i

        prompt = get_prompt(item)
        target = get_target(item, target_key=target_key)

        if prompt in done_prompts:
            print(f"[SKIP] index={global_i} already done.")
            continue

        print("\n" + "=" * 80)
        print(f"[*] Running item {global_i}")
        print(f"[*] Target key: {target_key}")
        print("=" * 80)

        # ============================================================
        # 构造 messages，与 Qwen2_7b 类保持一致（使用 apply_chat_template）
        # user content 中包含占位符 {optim_str}，供 GCG 替换
        # ============================================================
        messages = [
            {"role": "system", "content": qwen_sys},
            {"role": "user", "content": f"{prompt} {{optim_str}}"}
        ]

        # DEBUG 检查占位符是否存在
        print("\n[DEBUG] Messages:")
        print(f"  system: {messages[0]['content']}")
        print(f"  user: {messages[1]['content']}")
        print("[DEBUG] Contains {optim_str}:", "{optim_str}" in messages[1]["content"])
        print("[DEBUG] Target preview:")
        print(repr(target[:300]))
        print()

        try:
            # ============================================================
            # 运行 GCG 攻击
            # ============================================================
            result = nanogcg.run(
                model=raw_model,
                tokenizer=tokenizer,
                messages=messages,
                target=target,
                config=config,
            )

            best_suffix = result.best_string
            best_loss = result.best_loss

            print(f"\n[✓] Best loss: {best_loss}")
            print(f"[✓] Best suffix: {best_suffix}\n")

            # ============================================================
            # 最终生成：构造完整的输入（不含占位符）
            # 使用与 Qwen2_7b 完全相同的 apply_chat_template 方式
            # ============================================================
            eval_messages = [
                {"role": "system", "content": qwen_sys},
                {"role": "user", "content": f"{prompt} {best_suffix}"}
            ]
            eval_prompt = tokenizer.apply_chat_template(
                eval_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            print("\n[DEBUG] Eval formatted prompt preview:")
            print(repr(eval_prompt[:1000]))
            print()

            # Tokenize 并生成
            gen_tokenized = tokenizer(
                eval_prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(device)

            with torch.no_grad():
                gen_outputs = raw_model.generate(
                    input_ids=gen_tokenized.input_ids,
                    attention_mask=gen_tokenized.attention_mask,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response_tokens = gen_outputs[0][gen_tokenized.input_ids.shape[1]:]
            model_response = tokenizer.decode(
                response_tokens,
                skip_special_tokens=True,
            )

            # 保存记录
            record = {
                "index": global_i,
                "model_id": model_name,
                "prompt": prompt,
                "target_key": target_key,
                "target": target,
                "best_suffix": best_suffix,
                "best_loss": float(best_loss),
                "best_target_loss": (
                    float(result.best_target_loss)
                    if result.best_target_loss is not None else None
                ),
                "best_som_loss": (
                    float(result.best_som_loss)
                    if result.best_som_loss is not None else None
                ),
                "full_input": f"{prompt} {best_suffix}",
                "formatted_eval_prompt": eval_prompt,
                "model_response": model_response,
            }

            append_jsonl(output_path, record)

            print("\n--- [Model Response Under Attack] ---")
            print(model_response)
            print("-------------------------------------\n")
            print(f"[✓] Saved item {global_i} to {output_path}")

        except Exception as e:
            error_record = {
                "index": global_i,
                "model_id": model_name,
                "prompt": prompt,
                "target_key": target_key,
                "error": repr(e),
            }
            append_jsonl(output_path, error_record)
            print(f"[!] Error on item {global_i}: {repr(e)}")
            print(f"[!] Error record saved to {output_path}")

        finally:
            # 清理显存
            if "result" in locals():
                del result
            if "gen_tokenized" in locals():
                del gen_tokenized
            if "gen_outputs" in locals():
                del gen_outputs

            gc.collect()
            torch.cuda.empty_cache()

    print("\n[✓] Batch suffix generation completed.")

    # 清理模型
    del model_obj
    if "raw_model" in locals():
        del raw_model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()