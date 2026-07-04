import copy
import gc
import logging
import queue
import threading

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed
from scipy.stats import spearmanr

from nanogcg.utils import (
    INIT_CHARS,
    configure_pad_token,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
)

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ProbeSamplingConfig:
    draft_model: transformers.PreTrainedModel
    draft_tokenizer: transformers.PreTrainedTokenizer
    r: int = 8
    sampling_factor: int = 16


@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    probe_sampling_config: Optional[ProbeSamplingConfig] = None

    # SOM/IRIS residual activation hook loss.
    # True: compute SOM loss on IRIS-style residual activation sites:
    # pre-attn residual stream, attn write, post-attn residual stream,
    # MLP write, post-MLP residual stream.
    # False: fall back to the original hidden_states-based SOM loss.
    use_resid_write_hooks: bool = True

    # Extra multiplier applied to SOM loss after reduction.
    # Keep 1.0 first; tune only after checking SOM/Target ratio.
    som_loss_scale: float = 1.0

    enable_som_loss: bool = True


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]

    best_target_loss: float = None
    best_som_loss: float = None
    target_losses: List[float] = None
    som_losses: List[float] = None


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
            token ids
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])

    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )

    return torch.stack(filtered_ids)


class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)
        self.prefix_cache = None
        self.draft_prefix_cache = None

        self.stop_flag = False

        self.draft_model = None
        self.draft_tokenizer = None
        self.draft_embedding_layer = None
        if self.config.probe_sampling_config:
            self.draft_model = self.config.probe_sampling_config.draft_model
            self.draft_tokenizer = self.config.probe_sampling_config.draft_tokenizer
            self.draft_embedding_layer = self.draft_model.get_input_embeddings()
            if self.draft_tokenizer.pad_token is None:
                configure_pad_token(self.draft_tokenizer)

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        # ======================================================
        # [Only-GCG / SOM-GCG switch]
        # ======================================================
        self.enable_som_loss = bool(getattr(config, "enable_som_loss", True))

        # 给所有 SOM 相关属性默认值，保证 only-GCG 时后面代码不会报错
        self.som_layer = None
        self.iris_layers = []
        self.use_beta_loss = False
        self.beta = 0.0
        self.som_coef = 0.0
        self.som_token_mode = "last_input"

        self.som_directions = None
        self.best_k_indices = []

        self.use_resid_write_hooks = False
        self.som_loss_scale = 1.0

        self._som_hook_enabled = False
        self._som_hook_positions = []
        self._som_hook_losses = []
        self._som_hook_handles = []

        if not self.enable_som_loss:
            logger.info("[✓] SOM loss disabled. Running template-aligned only-GCG.")
            return
        # ======================================================
        # [SOM + IRIS]: 初始化并加载 SOM 多拒绝方向
        # ======================================================
        # SOM 论文报告的 layer number 是 0-indexed；Llama2-7B 使用 layer 13/32。
        self.som_layer = 13

        # IRIS-style penalty：不要只在一个 layer 上惩罚。
        # 你可以先用这一组，之后再扫更密集的层集合。
        self.iris_layers = list(range(1, 33))

        # 显式 SOM loss 系数。先扫: 10, 30, 50, 100。
        self.use_beta_loss = True
        self.beta = 0.1
        self.som_coef = 0.0  # beta 模式下不用它，只保留给打印兼容

        # token 惩罚模式：
        #   "last_input"：最接近 IRIS，惩罚 target 前的最后一个 input token。
        #   "suffix"    ：惩罚整个 adversarial suffix 区间。
        #   "both"      ：同时惩罚 last_input 与 suffix 区间。
        self.som_token_mode = "last_input"

        som_path = "/home/wuhengzhi/som-refusal-directions-main/runs/llama2-7b/generate_directions/centroid_to_som4_sigma0.33_layer13_directions.pt"
        logger.info(f"[*] Loading SOM steering directions from {som_path}...")
        raw_directions = torch.load(som_path, map_location=model.device, weights_only=True)

        # BO 选出的 top-k SOM directions。确认这组和当前 model / layer / template 对齐。
        self.best_k_indices = [10,6,8,13,14,15,9]
        raw_directions = raw_directions[self.best_k_indices]

        # 投影 loss 用 fp32 更稳；真正 matmul 时 hidden states 也会转 fp32。
        self.som_directions = raw_directions.to(model.device).float()
        self.som_directions = torch.nn.functional.normalize(self.som_directions, dim=-1)

        logger.info(
            f"[✓] Selected {len(self.best_k_indices)} SOM directions. "
            f"Shape: {self.som_directions.shape}; coef={self.som_coef}; "
            f"token_mode={self.som_token_mode}; layers={self.iris_layers}"
        )

        # Residual-write hook loss.
        # SOM weight ablation edits embed_tokens, attention o_proj, and MLP down_proj.
        # Therefore the most faithful IRIS-style activation loss should observe
        # o_proj/down_proj residual writes rather than only final layer hidden_states.
        self.use_resid_write_hooks = bool(getattr(config, "use_resid_write_hooks", True))
        self.som_loss_scale = float(getattr(config, "som_loss_scale", 1.0))

        self._som_hook_enabled = False
        self._som_hook_positions = []
        self._som_hook_losses = []
        self._som_hook_handles = []

        if self.use_resid_write_hooks:
            self._register_resid_write_hooks()
            logger.info(
                f"[✓] IRIS-style residual activation hooks registered: {len(self._som_hook_handles)} hooks "
                f"(pre/post residual streams + o_proj/down_proj writes), som_loss_scale={self.som_loss_scale}"
            )
        else:
            logger.info("[*] Residual activation hooks disabled; using hidden_states SOM loss.")
        # ======================================================


    def close(self) -> None:
        """Remove registered hooks to avoid accumulating duplicate hooks across batch runs."""
        for handle in getattr(self, "_som_hook_handles", []):
            try:
                handle.remove()
            except Exception:
                pass
        self._som_hook_handles = []
        self._som_hook_enabled = False
        self._som_hook_losses = []
        self._som_hook_positions = []

    def _get_som_token_positions(
        self,
        input_len: int,
        target_len: int,
        after_len: int,
    ) -> List[int]:
        """
        Compute token positions used by the SOM/IRIS penalty.

        input_embeds layout when prefix_cache is used:
            [optim_suffix, after_template, target]
        otherwise:
            [before_template, optim_suffix, after_template, target]

        prompt_end is the position right after the full input prompt and
        right before target tokens.
        """
        prompt_end = input_len - target_len
        token_positions = []

        if self.som_token_mode in ["last_input", "both"]:
            token_positions.append(prompt_end - 1)

        if self.som_token_mode in ["suffix", "both"]:
            optim_len = getattr(self, "optim_len", None)
            if optim_len is not None:
                suffix_start = prompt_end - after_len - optim_len
                suffix_end = prompt_end - after_len
                token_positions.extend(range(suffix_start, suffix_end))

        return sorted(set(p for p in token_positions if 0 <= p < input_len))

    def _make_som_activation_hook(self, source_name: str, use_input: bool = False):
        """
        Build a hook that captures one IRIS-style activation site.

        use_input=True is used for LayerNorm modules because, in pre-norm
        decoder blocks, the LayerNorm input is the residual stream state.
        """
        def hook(module, inputs, output):
            if not getattr(self, "_som_hook_enabled", False):
                return

            if use_input:
                if not inputs:
                    return
                act = inputs[0]
            else:
                act = output[0] if isinstance(output, tuple) else output

            self._capture_som_activation(act, source_name=source_name)

        return hook

    def _register_resid_write_hooks(self) -> None:
        """
        Register hooks on activation sites that best approximate:

            last input token × every layer × every residual activation

        For each decoder block, we capture five sites:
            1. pre-attention residual stream
               = input_layernorm input
            2. attention residual write
               = self_attn.o_proj output
            3. post-attention residual stream
               = post_attention_layernorm input
            4. MLP residual write
               = mlp.down_proj output
            5. post-MLP residual stream / layer output
               = decoder block output

        self.iris_layers follows HF hidden_states indexing:
            hidden_states[1] == layer 0 output
        Therefore module layer i corresponds to hidden-state index i + 1.
        """
        try:
            layers = self.model.model.layers
        except AttributeError:
            logger.warning("[!] Cannot find self.model.model.layers; residual hooks not registered.")
            return

        allowed_hidden_indices = set(self.iris_layers)

        for layer_i, block in enumerate(layers):
            hidden_idx = layer_i + 1
            if hidden_idx not in allowed_hidden_indices:
                continue

            # 1. pre-attention residual stream, before input_layernorm.
            if hasattr(block, "input_layernorm"):
                handle = block.input_layernorm.register_forward_hook(
                    self._make_som_activation_hook(
                        source_name=f"layer{layer_i}.pre_attn_resid",
                        use_input=True,
                    )
                )
                self._som_hook_handles.append(handle)

            # 2. attention residual write.
            if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
                handle = block.self_attn.o_proj.register_forward_hook(
                    self._make_som_activation_hook(
                        source_name=f"layer{layer_i}.attn_write",
                        use_input=False,
                    )
                )
                self._som_hook_handles.append(handle)

            # 3. post-attention residual stream, before post_attention_layernorm.
            if hasattr(block, "post_attention_layernorm"):
                handle = block.post_attention_layernorm.register_forward_hook(
                    self._make_som_activation_hook(
                        source_name=f"layer{layer_i}.post_attn_resid",
                        use_input=True,
                    )
                )
                self._som_hook_handles.append(handle)

            # 4. MLP residual write.
            if hasattr(block, "mlp") and hasattr(block.mlp, "down_proj"):
                handle = block.mlp.down_proj.register_forward_hook(
                    self._make_som_activation_hook(
                        source_name=f"layer{layer_i}.mlp_write",
                        use_input=False,
                    )
                )
                self._som_hook_handles.append(handle)

            # 5. post-MLP residual stream / decoder block output.
            handle = block.register_forward_hook(
                self._make_som_activation_hook(
                    source_name=f"layer{layer_i}.post_mlp_resid",
                    use_input=False,
                )
            )
            self._som_hook_handles.append(handle)

    def _capture_som_activation(self, act, source_name: str = "") -> None:
        """
        Compute SOM projection loss for one activation tensor.

        act should be [B, seq, d_model]. We store only [B] loss tensors.
        Do not detach: compute_token_gradient needs gradients through this loss.
        """
        if act is None:
            return

        if not torch.is_tensor(act) or act.dim() != 3:
            return

        if not self._som_hook_positions:
            return

        dirs = self.som_directions.float().to(act.device)
        dirs = torch.nn.functional.normalize(dirs, dim=-1)

        pos_losses = []
        for pos in self._som_hook_positions:
            if 0 <= pos < act.shape[1]:
                h = act[:, pos, :].float()          # [B, d]
                proj = h @ dirs.T                   # [B, K]
                pos_losses.append(proj.pow(2).mean(dim=-1))  # [B]

        if pos_losses:
            self._som_hook_losses.append(torch.stack(pos_losses, dim=0).mean(dim=0))

    def _start_som_hooks(
        self,
        input_len: int,
        target_len: int,
        after_len: int,
    ) -> None:
        self._som_hook_positions = self._get_som_token_positions(
            input_len=input_len,
            target_len=target_len,
            after_len=after_len,
        )
        self._som_hook_losses = []
        self._som_hook_enabled = True

    def _finish_som_hooks(self, batch_size: int) -> Tensor:
        self._som_hook_enabled = False

        if not self._som_hook_losses:
            return torch.zeros(batch_size, device=self.model.device, dtype=torch.float32)

        som_loss = torch.stack(self._som_hook_losses, dim=0).mean(dim=0)
        return som_loss * self.som_loss_scale

    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")
        #print("\n[DEBUG] GCG template:")
        #print(repr(template[:1000]))
        #print("[DEBUG] before_str tail:", repr(before_str[-300:]))
        #print("[DEBUG] after_str:", repr(after_str))
        target = " " + target if config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True,output_hidden_states=self.enable_som_loss,)
                self.prefix_cache = output.past_key_values

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        # Initialize components for probe sampling, if enabled.
        if config.probe_sampling_config:
            assert self.draft_model and self.draft_tokenizer and self.draft_embedding_layer, "Draft model wasn't properly set up."

            # Tokenize everything that doesn't get optimized for the draft model
            draft_before_ids = self.draft_tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            draft_after_ids = self.draft_tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            self.draft_target_ids = self.draft_tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

            (
                self.draft_before_embeds,
                self.draft_after_embeds,
                self.draft_target_embeds,
            ) = [
                self.draft_embedding_layer(ids)
                for ids in (
                    draft_before_ids,
                    draft_after_ids,
                    self.draft_target_ids,
                )
            ]

            if config.use_prefix_cache:
                with torch.no_grad():
                    output = self.draft_model(inputs_embeds=self.draft_before_embeds, use_cache=True)
                    self.draft_prefix_cache = output.past_key_values

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()
        self.optim_len = optim_ids.shape[1]

        losses = []
        optim_strings = []

        target_losses = []
        som_losses = []

        for _ in tqdm(range(config.num_steps)):
            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():

                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)

                if self.config.probe_sampling_config is None:
                    loss = find_executable_batch_size(self._compute_candidates_loss_original,batch_size)(input_embeds)
                    current_loss = loss.min().item()
                    best_idx = loss.argmin()  # 🚨 提炼出本轮最优候选者的索引
                    current_target_loss = self.last_target_losses[best_idx].item()
                    current_som_loss = self.last_som_losses[best_idx].item()
                    optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
                    raw_target = self.last_target_losses[best_idx].item()
                    raw_som = self.last_som_losses[best_idx].item()
                    if self.use_beta_loss:
                        weighted_target = (1.0 - self.beta) * raw_target
                        weighted_som = self.beta * raw_som
                    else:
                        weighted_target = raw_target
                        weighted_som = self.som_coef * raw_som
                    ratio = weighted_som / (weighted_target + 1e-8)

                    print(
                        f"🔥 [Optimization Step] | "
                        f"Total Loss: {current_loss:.4f} | "
                        f"🎯 Target raw: {raw_target:.4f} | "
                        f"Target weighted: {weighted_target:.4f} | "
                        f"🛡️ SOM raw: {raw_som:.6f} | "
                        f"SOM weighted: {weighted_som:.6f} | "
                        f"SOM/Target ratio: {ratio:.6f}"
                    )
                else:
                    current_loss, optim_ids = find_executable_batch_size(self._compute_candidates_loss_probe_sampling, batch_size)(
                        input_embeds, sampled_ids,
                    )

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            target_losses.append(current_target_loss)
            som_losses.append(current_som_loss)
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            best_target_loss=target_losses[min_loss_index],
            best_som_loss=som_losses[min_loss_index],
            target_losses=target_losses,
            som_losses=som_losses,
        )

        return result

    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids

        else:  # assume list
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)

        init_buffer_losses = find_executable_batch_size(self._compute_candidates_loss_original, true_buffer_size)(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        return buffer

    def compute_som_iris_loss(
        self,
        hidden_states_pool,
        input_len: int,
        target_len: int,
        after_len: int,
        batch_size: int,
    ) -> Tensor:
        """
        IRIS-style SOM refusal suppression loss.

        IRIS penalizes hidden activations' projection onto a refusal direction.
        Here we replace the single refusal direction with BO-selected SOM
        directions and apply the penalty on multiple layers and selected
        input-token positions.

        Returns:
            Tensor of shape [batch_size], one SOM penalty per candidate.
        """
        if hidden_states_pool is None or len(hidden_states_pool) == 0:
            return torch.zeros(batch_size, device=self.model.device, dtype=torch.float32)

        dirs = self.som_directions.float()
        dirs = torch.nn.functional.normalize(dirs, dim=-1)

        token_positions = self._get_som_token_positions(
            input_len=input_len,
            target_len=target_len,
            after_len=after_len,
        )
        if not token_positions:
            return torch.zeros(batch_size, device=self.model.device, dtype=torch.float32)

        losses = []
        for layer_idx in self.iris_layers:
            if layer_idx >= len(hidden_states_pool):
                continue

            h_layer = hidden_states_pool[layer_idx].float()  # [B, seq, d]
            for pos in token_positions:
                h = h_layer[:, pos, :]      # [B, d]
                proj = h @ dirs.T           # [B, K]
                losses.append(proj.pow(2).mean(dim=-1))

        if not losses:
            return torch.zeros(batch_size, device=self.model.device, dtype=torch.float32)

        return torch.stack(losses, dim=0).mean(dim=0) * self.som_loss_scale

    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        #"""Compute gradient of target CE + optional SOM-IRIS penalty w.r.t. one-hot suffix tokens."""
        model = self.model
        embedding_layer = self.embedding_layer

        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids,
            num_classes=embedding_layer.num_embeddings,
        )
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat(
                [optim_embeds, self.after_embeds, self.target_embeds],
                dim=1,
            )
        else:
            input_embeds = torch.cat(
                [self.before_embeds, optim_embeds, self.after_embeds, self.target_embeds],
                dim=1,
            )

        after_len = self.after_embeds.shape[1] if self.after_embeds is not None else 0

        if self.enable_som_loss and self.use_resid_write_hooks:
            self._start_som_hooks(
                input_len=input_embeds.shape[1],
                target_len=self.target_ids.shape[1],
                after_len=after_len,
            )

        if self.prefix_cache:
            output = model(
                inputs_embeds=input_embeds,
                past_key_values=self.prefix_cache,
                use_cache=True,
                output_hidden_states=self.enable_som_loss,
            )
        else:
            output = model(
                inputs_embeds=input_embeds,
                output_hidden_states=self.enable_som_loss,
            )

        logits = output.logits

        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift - 1: -1, :].contiguous()
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(
                shift_logits,
                -1,
                shift_labels.unsqueeze(-1),
            ).squeeze(-1)
            target_loss = mellowmax(
                -label_logits,
                alpha=self.config.mellowmax_alpha,
                dim=-1,
            )
        else:
            target_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if self.enable_som_loss:
            if self.use_resid_write_hooks:
                som_loss = self._finish_som_hooks(batch_size=input_embeds.shape[0])
            else:
                hidden_states_pool = getattr(output, "hidden_states", None)
                som_loss = self.compute_som_iris_loss(
                    hidden_states_pool=hidden_states_pool,
                    input_len=input_embeds.shape[1],
                    target_len=self.target_ids.shape[1],
                    after_len=after_len,
                    batch_size=input_embeds.shape[0],
                )

            if self.use_beta_loss:
                total_gradient_loss = (1.0 - self.beta) * target_loss + self.beta * som_loss.mean()
            else:
                total_gradient_loss = target_loss + self.som_coef * som_loss.mean()
        else:
            total_gradient_loss = target_loss

        optim_ids_onehot_grad = torch.autograd.grad(
            outputs=[total_gradient_loss],
            inputs=[optim_ids_onehot],
        )[0]

        return optim_ids_onehot_grad

    def _compute_candidates_loss_original(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
    ) -> Tensor:
        """Compute target CE + optional SOM-IRIS loss for all candidate suffixes."""
        all_loss = []
        all_target_loss = []
        all_som_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                after_len = self.after_embeds.shape[1] if self.after_embeds is not None else 0

                if self.enable_som_loss and self.use_resid_write_hooks:
                    self._start_som_hooks(
                        input_len=input_embeds_batch.shape[1],
                        target_len=self.target_ids.shape[1],
                        after_len=after_len,
                    )

                if self.prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                        prefix_cache_batch = [
                            [x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[layer_i]]
                            for layer_i in range(len(self.prefix_cache))
                        ]

                    outputs = self.model(
                        inputs_embeds=input_embeds_batch,
                        past_key_values=prefix_cache_batch,
                        use_cache=True,
                        output_hidden_states=self.enable_som_loss,
                    )
                else:
                    outputs = self.model(
                        inputs_embeds=input_embeds_batch,
                        output_hidden_states=self.enable_som_loss,
                    )

                logits = outputs.logits

                tmp = input_embeds_batch.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp - 1:-1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)

                if self.config.use_mellowmax:
                    label_logits = torch.gather(
                        shift_logits,
                        -1,
                        shift_labels.unsqueeze(-1),
                    ).squeeze(-1)
                    target_loss = mellowmax(
                        -label_logits,
                        alpha=self.config.mellowmax_alpha,
                        dim=-1,
                    )
                else:
                    target_loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                    ).view(current_batch_size, -1).mean(dim=-1)

                if self.enable_som_loss:
                    if self.use_resid_write_hooks:
                        som_loss = self._finish_som_hooks(batch_size=current_batch_size)
                    else:
                        hidden_states_pool = getattr(outputs, "hidden_states", None)
                        som_loss = self.compute_som_iris_loss(
                            hidden_states_pool=hidden_states_pool,
                            input_len=input_embeds_batch.shape[1],
                            target_len=self.target_ids.shape[1],
                            after_len=after_len,
                            batch_size=current_batch_size,
                        )

                    if self.use_beta_loss:
                        loss = (1.0 - self.beta) * target_loss + self.beta * som_loss
                    else:
                        loss = target_loss + self.som_coef * som_loss
                else:
                    som_loss = torch.zeros(
                        current_batch_size,
                        device=input_embeds_batch.device,
                        dtype=torch.float32,
                    )
                    loss = target_loss

                all_loss.append(loss)
                all_target_loss.append(target_loss)
                all_som_loss.append(som_loss)

                if self.config.early_stop:
                    if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                        self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        self.last_target_losses = torch.cat(all_target_loss, dim=0)
        self.last_som_losses = torch.cat(all_som_loss, dim=0)

        return torch.cat(all_loss, dim=0)

    def _compute_candidates_loss_probe_sampling(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        sampled_ids: Tensor,
    ) -> Tuple[float, Tensor]:
        """Computes the GCG loss using probe sampling (https://arxiv.org/abs/2403.01251).

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
            sampled_ids: Tensor, all candidate token id sequences. Used for further sampling.

        Returns:
            A tuple of (min_loss: float, corresponding_sequence: Tensor)

        """
        probe_sampling_config = self.config.probe_sampling_config
        assert probe_sampling_config, "Probe sampling config wasn't set up properly."

        B = input_embeds.shape[0]
        probe_size = B // probe_sampling_config.sampling_factor
        probe_idxs = torch.randperm(B)[:probe_size].to(input_embeds.device)
        probe_embeds = input_embeds[probe_idxs]

        def _compute_probe_losses(result_queue: queue.Queue, search_batch_size: int, probe_embeds: Tensor) -> None:
            probe_losses = self._compute_candidates_loss_original(search_batch_size, probe_embeds)
            result_queue.put(("probe", probe_losses))

        def _compute_draft_losses(
            result_queue: queue.Queue,
            search_batch_size: int,
            draft_sampled_ids: Tensor,
        ) -> None:
            assert self.draft_model and self.draft_embedding_layer, "Draft model and embedding layer weren't initialized properly."

            draft_losses = []
            draft_prefix_cache_batch = None
            for i in range(0, B, search_batch_size):
                with torch.no_grad():
                    batch_size = min(search_batch_size, B - i)
                    draft_sampled_ids_batch = draft_sampled_ids[i : i + batch_size]

                    if self.draft_prefix_cache:
                        if not draft_prefix_cache_batch or batch_size != search_batch_size:
                            draft_prefix_cache_batch = [
                                [x.expand(batch_size, -1, -1, -1) for x in self.draft_prefix_cache[i]] for i in range(len(self.draft_prefix_cache))
                            ]
                        draft_embeds = torch.cat(
                            [
                                self.draft_embedding_layer(draft_sampled_ids_batch),
                                self.draft_after_embeds.repeat(batch_size, 1, 1),
                                self.draft_target_embeds.repeat(batch_size, 1, 1),
                            ],
                            dim=1,
                        )
                        draft_output = self.draft_model(
                            inputs_embeds=draft_embeds,
                            past_key_values=draft_prefix_cache_batch,
                        )
                    else:
                        draft_embeds = torch.cat(
                            [
                                self.draft_before_embeds.repeat(batch_size, 1, 1),
                                self.draft_embedding_layer(draft_sampled_ids_batch),
                                self.draft_after_embeds.repeat(batch_size, 1, 1),
                                self.draft_target_embeds.repeat(batch_size, 1, 1),
                            ],
                            dim=1,
                        )
                        draft_output = self.draft_model(inputs_embeds=draft_embeds)

                    draft_logits = draft_output.logits
                    tmp = draft_embeds.shape[1] - self.draft_target_ids.shape[1]
                    shift_logits = draft_logits[..., tmp - 1 : -1, :].contiguous()
                    shift_labels = self.draft_target_ids.repeat(batch_size, 1)

                    if self.config.use_mellowmax:
                        label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                        loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                    else:
                        loss = (
                            torch.nn.functional.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                reduction="none",
                            )
                            .view(batch_size, -1)
                            .mean(dim=-1)
                        )

                    draft_losses.append(loss)

            draft_losses = torch.cat(draft_losses)
            result_queue.put(("draft", draft_losses))

        def _convert_to_draft_tokens(token_ids: Tensor) -> Tensor:
            decoded_text_list = self.tokenizer.batch_decode(token_ids)
            assert self.draft_tokenizer, "Draft tokenizer wasn't properly initialized."
            return self.draft_tokenizer(
                decoded_text_list,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )[
                "input_ids"
            ].to(self.draft_model.device, torch.int64)

        result_queue = queue.Queue()
        draft_sampled_ids = _convert_to_draft_tokens(sampled_ids)

        # Step 1. Compute loss of all candidates using the draft model
        draft_thread = threading.Thread(
            target=_compute_draft_losses,
            args=(result_queue, search_batch_size, draft_sampled_ids),
        )

        # Step 2. In parallel to 1., compute loss of the probe set on target model
        probe_thread = threading.Thread(
            target=_compute_probe_losses,
            args=(result_queue, search_batch_size, probe_embeds),
        )

        draft_thread.start()
        probe_thread.start()

        draft_thread.join()
        probe_thread.join()

        results = {}
        while not result_queue.empty():
            key, value = result_queue.get()
            results[key] = value

        probe_losses = results["probe"]
        draft_losses = results["draft"]

        # Step 3. Calculate agreement score using Spearman correlation
        draft_probe_losses = draft_losses[probe_idxs]
        rank_correlation = spearmanr(
            probe_losses.cpu().type(torch.float32).numpy(),
            draft_probe_losses.cpu().type(torch.float32).numpy(),
        ).correlation
        # normalized from [-1, 1] to [0, 1]
        alpha = (1 + rank_correlation) / 2

        # Step 4. Calculate the filtered set and evaluate using the target model.
        R = probe_sampling_config.r
        filtered_size = int((1 - alpha) * B / R)
        filtered_size = max(1, min(filtered_size, B))

        _, top_indices = torch.topk(draft_losses, k=filtered_size, largest=False)

        filtered_embeds = input_embeds[top_indices]
        filtered_losses = self._compute_candidates_loss_original(search_batch_size, filtered_embeds)

        # Step 5. Return best loss between probe set and filtered set
        best_probe_loss = probe_losses.min().item()
        best_filtered_loss = filtered_losses.min().item()

        probe_ids = sampled_ids[probe_idxs]
        filtered_ids = sampled_ids[top_indices]
        return (
            (best_probe_loss, probe_ids[probe_losses.argmin()].unsqueeze(0))
            if best_probe_loss < best_filtered_loss
            else (
                best_filtered_loss,
                filtered_ids[filtered_losses.argmin()].unsqueeze(0),
            )
        )


# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Generates a single optimized string using GCG.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = GCG(model, tokenizer, config)
    try:
        result = gcg.run(messages, target)
    finally:
        gcg.close()
    return result
