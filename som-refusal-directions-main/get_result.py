import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    trust_remote_code=True
)

input_path = "runs/qwen2-7b/val_optim_completions/weights_raw_dirs_centroid_to_som4_sigma0.33_layer18_harmbench_val_8_2_10_7_14_11_3_harmbench_val_completions.json"
output_path = "targets/qwen2_som_ablation_targets.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def make_target(response, n_tokens):
    ids = tokenizer.encode(response, add_special_tokens=False)
    return tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)

new_data = []

for item in data:
    response = item["response"]

    item = dict(item)
    item["target_16"] = make_target(response, 16)
    item["target_32"] = make_target(response, 32)
    item["target_64"] = make_target(response, 64)

    new_data.append(item)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print(f"Saved to {output_path}")