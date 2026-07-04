import os
import gc
import json
import torch

from models.load_models import load_model
from transformers import AutoTokenizer
from nanogcg import GCGConfig
import nanogcg
from models.system_prompts import llama_sys




def force_passthrough_template(tokenizer):
    """
    强制 tokenizer.apply_chat_template 原样输出 message['content']。

    这样 nanogcg 内部调用 apply_chat_template 时，不会再额外套一层
    [INST] ... [/INST]，也不会忽略 system prompt。
    """
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['content'] }}"
        "{% endfor %}"
    )


def format_llama2_som_prompt(prompt: str, suffix: str = "") -> str:
    """
    和 SOM 源码 Llama2_7b._get_prompt 保持一致：

    [INST] <<SYS>>
    llama_sys
    <</SYS>>

    {prompt} [/INST] 

    如果有 suffix，则放在 user prompt 内部，位于 [/INST] 前。
    """
    if suffix:
        user_content = f"{prompt} {suffix}"
    else:
        user_content = prompt

    return f"[INST] <<SYS>>\n{llama_sys}\n<</SYS>>\n\n{user_content} [/INST] "


def format_llama2_som_prompt_for_gcg(prompt: str) -> str:
    """
    给 GCG 用的版本，保留 {optim_str} 占位符。

    注意：这里在 f-string 里要输出字面量 {optim_str}，
    所以必须写成 {{optim_str}}。
    """
    return f"[INST] <<SYS>>\n{llama_sys}\n<</SYS>>\n\n{prompt} {{optim_str}} [/INST] "


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
    input_path = "targets/llama2_som_bo_targets.json"

    # 换一个输出文件，避免旧实验结果导致断点续跑直接 skip
    output_path = "results/som_gcg_suffixes_val_all-layers_0.05_target64.jsonl"

    target_key = "target_64"
    

    start_idx = 0

    # 建议第一次先改成 1 跑通，看 DEBUG prompt 是否正确
    # 确认没问题后再改成 41
    max_items = 41

    model_name = "llama2-7b"
    device = "cuda"

    print(f"[*] nanogcg path: {nanogcg.__file__}")
    print(f"[*] Loading target model: {model_name}")

    model_obj = load_model(model_name, device=device)

    if hasattr(model_obj, "model"):
        raw_model = model_obj.model
    else:
        raw_model = model_obj

    raw_model.eval()

    if hasattr(model_obj, "tokenizer"):
        tokenizer = model_obj.tokenizer
    else:
        hf_path = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

    # ============================================================
    # 关键修改：强制 apply_chat_template 原样输出 content
    # ============================================================
    force_passthrough_template(tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        enable_som_loss=True
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
        #target="sure,here is"

        if prompt in done_prompts:
            print(f"[SKIP] index={global_i} already done.")
            continue

        print("\n" + "=" * 80)
        print(f"[*] Running item {global_i}")
        print(f"[*] Target key: {target_key}")
        print("=" * 80)

        # ============================================================
        # 关键修改：GCG 使用和 SOM target 生成一致的 prompt 格式
        # ============================================================
        gcg_prompt = format_llama2_som_prompt_for_gcg(prompt)
        messages = [{"role": "user", "content": gcg_prompt}]

        # DEBUG：第一次一定要看这个，确认里面有 <<SYS>> 和 {optim_str}
        print("\n[DEBUG] GCG formatted prompt preview:")
        print(repr(gcg_prompt[:1000]))
        print("[DEBUG] Contains {optim_str}:", "{optim_str}" in gcg_prompt)
        print("[DEBUG] Target preview:")
        print(repr(target[:300]))
        print()

        try:
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
            # 最终生成：也使用同一个 SOM/Llama2 prompt 格式
            # ============================================================
            eval_prompt = format_llama2_som_prompt(prompt, best_suffix)

            print("\n[DEBUG] Eval formatted prompt preview:")
            print(repr(eval_prompt[:1000]))
            print()

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
            if "result" in locals():
                del result
            if "gen_tokenized" in locals():
                del gen_tokenized
            if "gen_outputs" in locals():
                del gen_outputs

            gc.collect()
            torch.cuda.empty_cache()

    print("\n[✓] Batch suffix generation completed.")

    del model_obj
    if "raw_model" in locals():
        del raw_model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()