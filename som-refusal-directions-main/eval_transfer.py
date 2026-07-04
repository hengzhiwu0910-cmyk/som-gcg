import argparse
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, "
    "dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive "
    "in nature. If a question does not make any sense, or is not factually coherent, explain why instead "
    "of answering something not correct. If you don’t know the answer to a question, please don’t share "
    "false information."
)


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["data", "records", "items", "examples"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]

    raise ValueError(f"Unsupported json format: {path}")


def parse_name_path(items: List[str]) -> Dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        result[name.strip()] = path.strip()
    return result


def get_first(record: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return default


def load_suffix_records(path: str, method_name: str, max_num=None) -> List[Dict[str, Any]]:
    raw = load_json_or_jsonl(path)
    if max_num is not None:
        raw = raw[:max_num]

    records = []
    for i, r in enumerate(raw):
        suffix = get_first(r, ["best_suffix", "suffix", "best_string", "optim_str", "adv_suffix"])
        if suffix is None:
            raise KeyError(f"No suffix field found in suffix record {i}")

        instruction = get_first(r, ["instruction", "prompt", "behavior", "query"], "")

        records.append({
            "method": method_name,
            "source_index": get_first(r, ["index", "id", "source_index"], i),
            "source_instruction": instruction,
            "source_category": get_first(r, ["category", "source_category"], None),
            "suffix": suffix,
        })
    return records


def load_test_records(path: str, max_num=None, categories=None) -> List[Dict[str, Any]]:
    raw = load_json_or_jsonl(path)
    records = []

    for i, r in enumerate(raw):
        instruction = get_first(r, ["instruction", "prompt", "behavior", "query"])
        if instruction is None:
            raise KeyError(f"No instruction field found in test record {i}")

        cat = get_first(r, ["category"], None)
        if categories is not None and cat not in categories:
            continue

        records.append({
            "test_index": get_first(r, ["index", "id"], i),
            "test_instruction": instruction,
            "test_category": cat,
        })

    if max_num is not None:
        records = records[:max_num]

    return records


def uid_for(*parts) -> str:
    s = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def append_jsonl(path: str, record: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_finished_uids(path: str) -> set:
    finished = set()
    if not os.path.exists(path):
        return finished

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if "uid" in r:
                    finished.add(r["uid"])
            except Exception:
                pass
    return finished


def join_prompt_suffix(prompt: str, suffix: str) -> str:
    prompt = str(prompt).rstrip()
    suffix = str(suffix).strip()
    if suffix == "":
        return prompt
    return prompt + " " + suffix


def build_prompt_text(tokenizer, instruction: str, suffix: str, use_system_prompt: bool):
    user_content = join_prompt_suffix(instruction, suffix)

    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if use_system_prompt:
        return DEFAULT_SYSTEM_PROMPT + "\n\nUser: " + user_content + "\nAssistant:"
    return "User: " + user_content + "\nAssistant:"


def build_tasks(
    suffix_records_by_method: Dict[str, List[Dict[str, Any]]],
    test_records: List[Dict[str, Any]],
    model_specs: Dict[str, str],
    source_model_name: str,
    modes: List[str],
    pairing: str,
    max_test_per_suffix: int,
    include_source_in_cross: bool,
):
    tasks = []

    cross_models = {
        name: path for name, path in model_specs.items()
        if include_source_in_cross or name != source_model_name
    }

    if source_model_name not in model_specs:
        raise ValueError(f"source_model_name={source_model_name} not found in model_specs")

    source_only = {source_model_name: model_specs[source_model_name]}

    for method, suffix_records in suffix_records_by_method.items():

        if "same_prompt_cross_model" in modes:
            for s in suffix_records:
                if not s["source_instruction"]:
                    continue

                for model_name, model_path in cross_models.items():
                    uid = uid_for("same_prompt_cross_model", method, model_name, s["source_index"])
                    tasks.append({
                        "uid": uid,
                        "mode": "same_prompt_cross_model",
                        "method": method,
                        "model_name": model_name,
                        "model_path": model_path,
                        "source_index": s["source_index"],
                        "source_category": s["source_category"],
                        "eval_index": s["source_index"],
                        "eval_category": s["source_category"],
                        "instruction": s["source_instruction"],
                        "suffix": s["suffix"],
                    })

        if "diff_prompt_same_model" in modes:
            for s, t in pair_suffix_test(suffix_records, test_records, pairing, max_test_per_suffix):
                for model_name, model_path in source_only.items():
                    uid = uid_for("diff_prompt_same_model", method, model_name, s["source_index"], t["test_index"])
                    tasks.append({
                        "uid": uid,
                        "mode": "diff_prompt_same_model",
                        "method": method,
                        "model_name": model_name,
                        "model_path": model_path,
                        "source_index": s["source_index"],
                        "source_category": s["source_category"],
                        "eval_index": t["test_index"],
                        "eval_category": t["test_category"],
                        "instruction": t["test_instruction"],
                        "suffix": s["suffix"],
                    })

        if "diff_prompt_cross_model" in modes:
            for s, t in pair_suffix_test(suffix_records, test_records, pairing, max_test_per_suffix):
                for model_name, model_path in cross_models.items():
                    uid = uid_for("diff_prompt_cross_model", method, model_name, s["source_index"], t["test_index"])
                    tasks.append({
                        "uid": uid,
                        "mode": "diff_prompt_cross_model",
                        "method": method,
                        "model_name": model_name,
                        "model_path": model_path,
                        "source_index": s["source_index"],
                        "source_category": s["source_category"],
                        "eval_index": t["test_index"],
                        "eval_category": t["test_category"],
                        "instruction": t["test_instruction"],
                        "suffix": s["suffix"],
                    })

    return tasks


def pair_suffix_test(
    suffix_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    pairing: str,
    max_test_per_suffix: int,
):
    if pairing == "same_index":
        for i, s in enumerate(suffix_records):
            if i < len(test_records):
                yield s, test_records[i]
        return

    if pairing == "round_robin":
        for i, s in enumerate(suffix_records):
            yield s, test_records[i % len(test_records)]
        return

    if pairing == "category":
        by_cat = {}
        for t in test_records:
            by_cat.setdefault(t["test_category"], []).append(t)

        for s in suffix_records:
            candidates = by_cat.get(s["source_category"], test_records)
            for t in candidates[:max_test_per_suffix]:
                yield s, t
        return

    if pairing == "all":
        for s in suffix_records:
            for t in test_records[:max_test_per_suffix]:
                yield s, t
        return

    raise ValueError(f"Unknown pairing: {pairing}")


def load_model(model_path: str, dtype: str, device_map: str, trust_remote_code: bool):
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    return model, tokenizer


def generate_for_model(model_name: str, model_path: str, tasks: List[Dict[str, Any]], args):
    print(f"\n[Load model] {model_name}: {model_path}")

    model, tokenizer = load_model(
        model_path=model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    device = next(model.parameters()).device

    for start in tqdm(range(0, len(tasks), args.batch_size), desc=model_name):
        batch_tasks = tasks[start:start + args.batch_size]

        prompt_texts = [
            build_prompt_text(
                tokenizer=tokenizer,
                instruction=t["instruction"],
                suffix=t["suffix"],
                use_system_prompt=args.use_system_prompt,
            )
            for t in batch_tasks
        ]

        inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(device)

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if args.do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        response_ids = output_ids[:, input_len:]
        responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        for task, prompt_text, response in zip(batch_tasks, prompt_texts, responses):
            rec = dict(task)
            rec["use_system_prompt"] = args.use_system_prompt
            rec["response"] = response
            if args.save_prompt_text:
                rec["prompt_text"] = prompt_text
            append_jsonl(args.out, rec)

    del model
    del tokenizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--suffix_files", nargs="+", required=True,
                        help="Format: method=path. Example: only=xxx.jsonl som005=yyy.jsonl")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Format: model_name=model_path. Example: llama2-7b=/path/to/model")
    parser.add_argument("--source_model_name", required=True)

    parser.add_argument("--test_file",
                        default="/home/wuhengzhi/som-refusal-directions-main/dataset/processed/harmbench_test.json")

    parser.add_argument("--modes", nargs="+",
                        default=["same_prompt_cross_model", "diff_prompt_same_model", "diff_prompt_cross_model"],
                        choices=["same_prompt_cross_model", "diff_prompt_same_model", "diff_prompt_cross_model"])

    parser.add_argument("--pairing", default="same_index",
                        choices=["same_index", "round_robin", "category", "all"])

    parser.add_argument("--max_suffixes", type=int, default=None)
    parser.add_argument("--max_test_prompts", type=int, default=None)
    parser.add_argument("--max_test_per_suffix", type=int, default=1)
    parser.add_argument("--categories", nargs="*", default=None)

    parser.add_argument("--out", default="results/transfer_generations.jsonl")

    parser.add_argument("--use_system_prompt", action="store_true")
    parser.add_argument("--include_source_in_cross", action="store_true")
    parser.add_argument("--save_prompt_text", action="store_true")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    suffix_file_map = parse_name_path(args.suffix_files)
    model_specs = parse_name_path(args.models)

    suffix_records_by_method = {}
    for method, path in suffix_file_map.items():
        suffix_records_by_method[method] = load_suffix_records(
            path=path,
            method_name=method,
            max_num=args.max_suffixes,
        )
        print(f"[Suffix] {method}: {len(suffix_records_by_method[method])}")

    test_records = load_test_records(
        path=args.test_file,
        max_num=args.max_test_prompts,
        categories=set(args.categories) if args.categories else None,
    )
    print(f"[Test prompts] {len(test_records)}")

    tasks = build_tasks(
        suffix_records_by_method=suffix_records_by_method,
        test_records=test_records,
        model_specs=model_specs,
        source_model_name=args.source_model_name,
        modes=args.modes,
        pairing=args.pairing,
        max_test_per_suffix=args.max_test_per_suffix,
        include_source_in_cross=args.include_source_in_cross,
    )

    finished = load_finished_uids(args.out)
    tasks = [t for t in tasks if t["uid"] not in finished]

    print(f"[Tasks remaining] {len(tasks)}")
    print(f"[Output] {args.out}")

    tasks_by_model = {}
    for t in tasks:
        tasks_by_model.setdefault(t["model_name"], []).append(t)

    for model_name, model_tasks in tasks_by_model.items():
        model_path = model_tasks[0]["model_path"]
        generate_for_model(
            model_name=model_name,
            model_path=model_path,
            tasks=model_tasks,
            args=args,
        )

    print("\nDone.")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()