from __future__ import annotations

import copy
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
import torch.nn as nn

from .gemma4_transformers import (
    KVCache,
    RMSNorm,
    ScaledWordEmbedding,
    TransformerBlock,
    compute_rope_params,
)

DEFAULT_GEMMA4_E2B_REPO_ID = "google/gemma-4-E2B"
DEFAULT_GEMMA4_E2B_IT_REPO_ID = "google/gemma-4-E2B-it"


GEMMA4_E2B_LAYER_TYPES = [
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
]


GEMMA4_CONFIG_E2B = {
    "repo_id": DEFAULT_GEMMA4_E2B_REPO_ID,
    "vocab_size": 262_144,
    "context_length": 131_072,
    "emb_dim": 1_536,
    "n_heads": 8,
    "n_layers": 35,
    "hidden_dim": 6_144,
    "head_dim": 256,
    "global_head_dim": 512,
    "qk_norm": True,
    "n_kv_groups": 1,
    "num_global_kv_groups": None,
    "sliding_window": 512,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {
            "rope_theta": 10_000.0,
            "rope_type": "default",
        },
    },
    "rms_norm_eps": 1e-6,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size_per_layer_input": 256,
    "vocab_size_per_layer_input": 262_144,
    "num_kv_shared_layers": 20,
    "attention_k_eq_v": False,
    "enable_moe_block": False,
    "use_double_wide_mlp": True,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": True,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "bos_token_id": 2,
    "dtype": torch.bfloat16,
    "layer_types": list(GEMMA4_E2B_LAYER_TYPES),
}


def copy_model_config(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def build_gemma4_e2b_config(**overrides) -> dict:
    cfg = copy_model_config(GEMMA4_CONFIG_E2B)
    cfg.update(overrides)
    if "layer_types" in overrides:
        cfg["layer_types"] = list(overrides["layer_types"])
    return cfg


class Gemma4Model(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        cfg = copy_model_config(cfg)
        self._validate_config(cfg)
        self.cfg = cfg
        self.current_pos = 0

        pad_token_id = cfg.get("pad_token_id")
        self.tok_emb = ScaledWordEmbedding(
            cfg["vocab_size"],
            cfg["emb_dim"],
            padding_idx=pad_token_id,
            embed_scale=cfg["emb_dim"] ** 0.5,
            dtype=cfg["dtype"],
        )
        self.blocks = nn.ModuleList([TransformerBlock(cfg, idx) for idx in range(cfg["n_layers"])])
        self.trf_blocks = self.blocks
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        if cfg.get("tie_word_embeddings", True):
            self.out_head.weight = self.tok_emb.weight

        self.hidden_size_per_layer_input = int(cfg.get("hidden_size_per_layer_input", 0) or 0)
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = ScaledWordEmbedding(
                cfg["vocab_size_per_layer_input"],
                cfg["n_layers"] * self.hidden_size_per_layer_input,
                padding_idx=pad_token_id,
                embed_scale=self.hidden_size_per_layer_input ** 0.5,
                dtype=cfg["dtype"],
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = nn.Linear(
                cfg["emb_dim"],
                cfg["n_layers"] * self.hidden_size_per_layer_input,
                bias=False,
                dtype=cfg["dtype"],
            )
            self.per_layer_model_projection_scale = cfg["emb_dim"] ** -0.5
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                eps=cfg.get("rms_norm_eps", 1e-6),
            )

        rope_tables = {}
        for layer_type, rope_params in cfg["rope_parameters"].items():
            head_dim = (
                int(cfg.get("global_head_dim") or cfg["head_dim"])
                if layer_type == "full_attention"
                else int(cfg["head_dim"])
            )
            rope_tables[layer_type] = compute_rope_params(
                head_dim=head_dim,
                theta_base=float(rope_params["rope_theta"]),
                context_length=cfg["context_length"],
                rope_type=rope_params.get("rope_type", "default"),
                partial_rotary_factor=float(rope_params.get("partial_rotary_factor", 1.0)),
                dtype=torch.float32,
            )

        for layer_type, (cos, sin) in rope_tables.items():
            safe_name = layer_type.replace("_attention", "")
            self.register_buffer(f"cos_{safe_name}", cos, persistent=False)
            self.register_buffer(f"sin_{safe_name}", sin, persistent=False)

    @staticmethod
    def _validate_config(cfg: dict) -> None:
        layer_types = cfg.get("layer_types")
        if not layer_types or len(layer_types) != cfg["n_layers"]:
            raise ValueError("cfg['layer_types'] must contain exactly cfg['n_layers'] entries.")
        unsupported = set(layer_types) - {"sliding_attention", "full_attention"}
        if unsupported:
            raise ValueError(f"Unsupported Gemma 4 layer types: {sorted(unsupported)}")
        if cfg.get("enable_moe_block", False):
            raise NotImplementedError("Gemma 4 E2B is dense; MoE support is intentionally not included.")

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.hidden_size_per_layer_input:
            raise RuntimeError("This Gemma 4 config does not use per-layer embeddings.")
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.cfg["n_layers"],
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.hidden_size_per_layer_input:
            raise RuntimeError("This Gemma 4 config does not use per-layer embeddings.")

        per_layer_projection = (
            self.per_layer_model_projection(inputs_embeds)
            * self.per_layer_model_projection_scale
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.cfg["n_layers"],
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def _create_masks(
        self,
        *,
        pos_start: int,
        pos_end: int,
        device: torch.device,
        attn_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query_positions = torch.arange(pos_start, pos_end, device=device)
        key_positions = torch.arange(0, pos_end, device=device)

        future_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        masks = {"full_attention": future_mask}

        sliding_window = int(self.cfg["sliding_window"])
        far_past_mask = (query_positions.unsqueeze(1) - key_positions.unsqueeze(0)) >= sliding_window
        masks["sliding_attention"] = future_mask | far_past_mask

        for layer_type, mask in list(masks.items()):
            mask = mask.unsqueeze(0).unsqueeze(0)
            if attn_mask is not None:
                key_padding_mask = (~attn_mask[:, :pos_end]).view(attn_mask.shape[0], 1, 1, pos_end)
                mask = mask | key_padding_mask
            masks[layer_type] = mask
        return masks

    def _rope_for_layer_type(self, layer_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        safe_name = layer_type.replace("_attention", "")
        return getattr(self, f"cos_{safe_name}"), getattr(self, f"sin_{safe_name}")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if input_ids is not None:
            inputs_embeds = self.tok_emb(input_ids)
        elif self.hidden_size_per_layer_input and per_layer_inputs is None:
            raise ValueError("input_ids or per_layer_inputs is required when PLE is enabled.")

        assert inputs_embeds is not None
        batch_size, seq_len, _ = inputs_embeds.shape

        if attn_mask is not None:
            attn_mask = attn_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if attn_mask.shape[0] != batch_size:
                raise ValueError("attn_mask batch size must match the inputs.")

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                assert input_ids is not None
                per_layer_inputs = self.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + seq_len
            self.current_pos = pos_end
            shared_layers = cache.shared_layers
        else:
            pos_start = 0
            pos_end = seq_len
            shared_layers = {}

        masks = self._create_masks(
            pos_start=pos_start,
            pos_end=pos_end,
            device=inputs_embeds.device,
            attn_mask=attn_mask,
        )

        hidden_states = inputs_embeds
        for idx, block in enumerate(self.blocks):
            cos, sin = self._rope_for_layer_type(block.layer_type)
            layer_ple = per_layer_inputs[:, :, idx, :] if per_layer_inputs is not None else None
            hidden_states = block(
                hidden_states,
                per_layer_input=layer_ple,
                attention_mask=masks[block.layer_type],
                cos=cos,
                sin=sin,
                start_pos=pos_start,
                cache=cache,
                shared_layers=shared_layers,
            )

        hidden_states = self.final_norm(hidden_states)
        if return_hidden_states:
            return hidden_states

        logits = self.out_head(hidden_states.to(self.cfg["dtype"]))
        softcap = self.cfg.get("final_logit_softcapping")
        if softcap is not None:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        return logits

    def reset_kv_cache(self) -> None:
        self.current_pos = 0


def build_model(cfg: dict | None = None) -> Gemma4Model:
    return Gemma4Model(copy_model_config(cfg or GEMMA4_CONFIG_E2B))




def generate_text_simple(
    model: Gemma4Model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int | None = None,
    use_cache: bool = True,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    ctx_len = context_size or model.cfg["context_length"]

    with torch.no_grad():
        if use_cache:
            cache = KVCache(n_layers=model.cfg["n_layers"])
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], cache=cache)

            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, cache=cache)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], cache=None)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def generate_text_simple_stream(
    model: Gemma4Model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    context_size: int | None = None,
):
    model.eval()
    ctx_len = context_size or model.cfg["context_length"]

    with torch.no_grad():
        cache = KVCache(n_layers=model.cfg["n_layers"])
        model.reset_kv_cache()
        logits = model(token_ids[:, -ctx_len:], cache=cache)

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)
            logits = model(next_token, cache=cache)


def download_from_huggingface(
    repo_id: str,
    filename: str,
    local_dir: str | Path,
    revision: str = "main",
) -> str:
    base_url = "https://huggingface.co"
    url = f"{base_url}/{repo_id}/resolve/{revision}/{filename}"
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    dest_path = local_dir / filename

    if dest_path.exists():
        return str(dest_path)

    request = Request(url, headers={"User-Agent": "train-ava"})
    try:
        with urlopen(request, timeout=60) as response, open(dest_path, "wb") as file:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                file.write(chunk)
    except (HTTPError, URLError, TimeoutError) as exc:
        dest_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    return str(dest_path)


class Gemma4Tokenizer:
    def __init__(
        self,
        tokenizer_file_path: str | Path = "tokenizer.json",
        repo_id: str | None = DEFAULT_GEMMA4_E2B_IT_REPO_ID,
        apply_chat_template: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        add_bos_token: bool = True,
    ):
        from tokenizers import Tokenizer

        self.apply_chat_template_by_default = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.enable_thinking = enable_thinking
        self.add_bos_token = add_bos_token

        tok_file = Path(tokenizer_file_path)
        if not tok_file.is_file() and repo_id:
            download_from_huggingface(repo_id, tok_file.name, tok_file.parent)

        self._tok = Tokenizer.from_file(str(tok_file))

        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.sot_token = "<|turn>"
        self.eot_token = "<turn|>"
        self.soc_token = "<|channel>"
        self.eoc_token = "<channel|>"
        self.think_token = "<|think|>"

        self.bos_token_id = self._tok.token_to_id(self.bos_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        self.pad_token_id = self._tok.token_to_id(self.pad_token)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        chat_wrapped: bool | None = None,
    ) -> list[int]:
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template_by_default
        if chat_wrapped:
            text = self.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=self.add_generation_prompt,
                enable_thinking=self.enable_thinking,
            )

        ids = self._tok.encode(text, add_special_tokens=False).ids
        if (
            add_special_tokens
            and self.add_bos_token
            and self.bos_token_id is not None
            and (not ids or ids[0] != self.bos_token_id)
        ):
            ids = [self.bos_token_id, *ids]
        return ids

    def decode(self, ids: list[int] | torch.Tensor | int, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().reshape(-1).tolist()
        elif isinstance(ids, int):
            ids = [ids]
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool | None = None,
        enable_thinking: bool | None = None,
    ) -> str | list[int]:
        add_generation_prompt = (
            self.add_generation_prompt if add_generation_prompt is None else add_generation_prompt
        )
        enable_thinking = self.enable_thinking if enable_thinking is None else enable_thinking

        rendered = [self.bos_token]
        loop_messages = list(messages)

        has_system = bool(loop_messages and loop_messages[0].get("role") in {"system", "developer"})
        if enable_thinking or has_system:
            rendered.append(f"{self.sot_token}system\n")
            if enable_thinking:
                rendered.append(self.think_token)
            if has_system:
                rendered.append(str(loop_messages[0].get("content", "")).strip())
                loop_messages = loop_messages[1:]
            rendered.append(f"{self.eot_token}\n")

        for message in loop_messages:
            role = "model" if message.get("role") == "assistant" else str(message.get("role", "user"))
            rendered.append(f"{self.sot_token}{role}\n")
            content = self._format_content(message.get("content", ""))
            if role == "model":
                content = self._strip_thinking(content)
            rendered.append(content.strip())
            rendered.append(f"{self.eot_token}\n")

        if add_generation_prompt:
            rendered.append(f"{self.sot_token}model\n")

        text = "".join(rendered)
        if tokenize:
            return self.encode(text, add_special_tokens=False, chat_wrapped=False)
        return text

    @staticmethod
    def _format_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append(str(item.get("text", "")))
                    elif item_type == "image":
                        parts.append("\n\n<|image|>\n\n")
                    elif item_type == "audio":
                        parts.append("<|audio|>")
                    elif item_type == "video":
                        parts.append("\n\n<|video|>\n\n")
            return "".join(parts)
        return str(content)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<\|channel\>thought\n.*?<channel\|>", "", text, flags=re.DOTALL).strip()


def text_to_token_ids(
    text: str,
    tokenizer: Gemma4Tokenizer,
    *,
    device: torch.device | str | None = None,
    chat_wrapped: bool | None = None,
) -> torch.Tensor:
    encoded = tokenizer.encode(text, chat_wrapped=chat_wrapped)
    return torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor | list[int], tokenizer: Gemma4Tokenizer) -> str:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().reshape(-1).tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=False)
