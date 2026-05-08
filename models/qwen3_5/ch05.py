import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
import torch.nn as nn

from .qwen3_5_transformers import Qwen3_5GatedDeltaNet


QWEN3_5_CONFIG_0_8B = {
    "vocab_size": 248_320,
    "context_length": 262_144,
    "emb_dim": 1_024,
    "n_heads": 8,
    "n_layers": 24,
    "hidden_dim": 3_584,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-6,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "dtype": torch.bfloat16,
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ],
}

QWEN3_5_CONFIG_2B = {
    **QWEN3_5_CONFIG_0_8B,
    "emb_dim": 2_048,
    "hidden_dim": 6_144,
    "layer_types": list(QWEN3_5_CONFIG_0_8B["layer_types"]),
}


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)

 
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        x_norm = self._norm(x.float())
        x_norm = x_norm * (1.0 + self.weight.float())
        return x_norm.to(dtype=x.dtype)


def compute_rope_params(
    head_dim,
    theta_base=10_000,
    context_length=4096,
    partial_rotary_factor=1.0,
    dtype=torch.float32,
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    rotary_dim = int(head_dim * partial_rotary_factor)
    rotary_dim = max(2, rotary_dim - (rotary_dim % 2))

    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, rotary_dim, 2, dtype=dtype)[: (rotary_dim // 2)].float() / rotary_dim
        )
    )

    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    rot_dim = cos.shape[-1]
    if rot_dim > head_dim:
        raise ValueError(f"RoPE dim {rot_dim} cannot exceed head_dim {head_dim}.")

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    x1 = x_rot[..., : rot_dim // 2]
    x2 = x_rot[..., rot_dim // 2 :]

    cos = cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x_rot * cos) + (rotated * sin)
    x_out = torch.cat([x_rotated, x_pass], dim=-1)
    return x_out.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out * 2, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape

        q_and_gate = self.W_query(x)
        q_and_gate = q_and_gate.view(b, num_tokens, self.num_heads, self.head_dim * 2)
        queries, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(b, num_tokens, self.d_out)

        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        prev_len = 0
        if cache is not None:
            prev_k, prev_v = cache
            if prev_k is not None:
                prev_len = prev_k.size(2)
                keys_cat_raw = torch.cat([prev_k, keys_new], dim=2)
                values_cat_raw = torch.cat([prev_v, values_new], dim=2)
            else:
                keys_cat_raw = keys_new
                values_cat_raw = values_new
        else:
            keys_cat_raw = keys_new
            values_cat_raw = values_new

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys = apply_rope(keys_cat_raw, cos, sin, offset=start_pos - prev_len)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values_cat_raw.repeat_interleave(self.group_size, dim=1)

        if cache is not None and cache[0] is not None:
            next_cache = (
                torch.cat([cache[0], keys_new], dim=2),
                torch.cat([cache[1], values_new], dim=2),
            )
        else:
            next_cache = (keys_new, values_new)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores * (self.head_dim ** -0.5),
            dim=-1,
            dtype=torch.float32,
        ).to(queries.dtype)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        context = context * torch.sigmoid(gate)
        out = self.out_proj(context)
        return out, next_cache


class _Qwen3_5ConfigAdapter:
    def __init__(self, cfg):
        self.hidden_size = cfg["emb_dim"]
        self.linear_num_value_heads = cfg["linear_num_value_heads"]
        self.linear_num_key_heads = cfg["linear_num_key_heads"]
        self.linear_key_head_dim = cfg["linear_key_head_dim"]
        self.linear_value_head_dim = cfg["linear_value_head_dim"]
        self.linear_conv_kernel_dim = cfg["linear_conv_kernel_dim"]
        self.hidden_act = "silu"
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-6)
        self.dtype = cfg.get("dtype", None)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_type, layer_idx):
        super().__init__()
        self.layer_type = layer_type

        if layer_type == "full_attention":
            self.token_mixer = GroupedQueryAttention(
                d_in=cfg["emb_dim"],
                num_heads=cfg["n_heads"],
                head_dim=cfg["head_dim"],
                num_kv_groups=cfg["n_kv_groups"],
                qk_norm=cfg["qk_norm"],
                dtype=cfg["dtype"],
            )
        elif layer_type == "linear_attention":
            self.token_mixer = Qwen3_5GatedDeltaNet(_Qwen3_5ConfigAdapter(cfg), layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))

    def forward(
        self,
        x,
        mask,
        cos,
        sin,
        start_pos=0,
        cache=None,
        linear_cache=None,
        cache_position=None,
        attention_mask=None,
    ):
        shortcut = x
        x = self.norm1(x)

        if self.layer_type == "full_attention":
            x, next_cache = self.token_mixer(
                x,
                mask,
                cos,
                sin,
                start_pos=start_pos,
                cache=cache,
            )
        else:
            x = self.token_mixer(
                x,
                cache_params=linear_cache,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
            next_cache = None

        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x, next_cache


class Qwen3_5LinearAttentionCache:
    def __init__(self, n_layers):
        self.conv_states = [None] * n_layers
        self.recurrent_states = [None] * n_layers
        self.has_previous_state = False

    def reset(self):
        for i in range(len(self.conv_states)):
            self.conv_states[i] = None
            self.recurrent_states[i] = None
        self.has_previous_state = False


class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers
        self.linear_cache = Qwen3_5LinearAttentionCache(n_layers)

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None
        self.linear_cache.reset()


class Qwen3_5Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        layer_types = cfg.get("layer_types", ["full_attention"] * cfg["n_layers"])
        if len(layer_types) != cfg["n_layers"]:
            raise ValueError("len(layer_types) must equal n_layers")

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_type, idx) for idx, layer_type in enumerate(layer_types)]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["emb_dim"] // cfg["n_heads"] if cfg["head_dim"] is None else cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            partial_rotary_factor=cfg.get("partial_rotary_factor", 1.0),
            dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0

    def create_mask(self, cur_len, device, pos_start=0, pos_end=None, attn_mask=None):
        if pos_end is None:
            pos_end = cur_len

        ones = torch.ones((pos_end, pos_end), device=device, dtype=torch.bool)
        mask_full = torch.triu(ones, diagonal=1)
        row_slice = slice(pos_start, pos_end)
        mask = mask_full[row_slice, :pos_end][None, None, :, :]

        if attn_mask is None:
            return mask

        key_padding_mask = (~attn_mask[:, :pos_end]).view(attn_mask.shape[0], 1, 1, pos_end)
        return mask | key_padding_mask

    def forward(self, in_idx, cache=None, attn_mask=None, return_hidden_states=False):
        x = self.tok_emb(in_idx)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device=x.device, dtype=torch.bool)

        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = self.create_mask(
                cur_len=num_tokens,
                device=x.device,
                pos_start=pos_start,
                pos_end=pos_end,
                attn_mask=attn_mask,
            )
            cache_position = torch.arange(pos_start, pos_end, device=x.device, dtype=torch.long)
        else:
            pos_start = 0
            mask = self.create_mask(
                cur_len=num_tokens,
                device=x.device,
                pos_start=0,
                pos_end=num_tokens,
                attn_mask=attn_mask,
            )
            cache_position = None

        if attn_mask is not None:
            qmask = attn_mask[:, pos_start:pos_start + num_tokens].unsqueeze(-1)
            x = x * qmask.to(x.dtype)

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache is not None else None
            x, new_blk_cache = block(
                x,
                mask=mask,
                cos=self.cos,
                sin=self.sin,
                start_pos=pos_start,
                cache=blk_cache,
                linear_cache=cache.linear_cache if cache is not None else None,
                cache_position=cache_position,
                attention_mask=(
                    attn_mask[:, pos_start:pos_start + num_tokens]
                    if attn_mask is not None
                    else None
                ),
            )
            if cache is not None and new_blk_cache is not None:
                cache.update(i, new_blk_cache)

        if cache is not None:
            cache.linear_cache.has_previous_state = True

        x = self.final_norm(x)
        if return_hidden_states:
            return x
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0


def build_model(cfg):
    return Qwen3_5Model(cfg)


def generate_text_simple(
    model,
    idx,
    max_new_tokens,
    context_size=None,
    use_cache=True,
    eos_token_id=None,
):
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


def generate_text_simple_stream(model, token_ids, max_new_tokens, eos_token_id=None, context_size=None):
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


def download_from_huggingface(repo_id, filename, local_dir, revision="main"):
    base_url = "https://huggingface.co"
    url = f"{base_url}/{repo_id}/resolve/{revision}/{filename}"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    dest_path = os.path.join(local_dir, filename)

    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return dest_path

    print(f"Downloading {url} to {dest_path}...")
    request = Request(url, headers={"User-Agent": "train-ava"})
    try:
        with urlopen(request, timeout=60) as response, open(dest_path, "wb") as file:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                file.write(chunk)
    except (HTTPError, URLError, TimeoutError) as exc:
        Path(dest_path).unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    return dest_path


class Qwen3_5Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<think>", "</think>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")

    def __init__(
        self,
        tokenizer_file_path="tokenizer.json",
        repo_id=None,
        apply_chat_template=True,
        add_generation_prompt=False,
        add_thinking=False,
        thinking_template="tagged",
    ):
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking
        if thinking_template not in {"tagged", "reasoning"}:
            raise ValueError("thinking_template must be 'tagged' or 'reasoning'.")
        self.thinking_template = thinking_template

        tok_file = Path(tokenizer_file_path)
        if not tok_file.is_file() and repo_id:
            download_from_huggingface(
                repo_id=repo_id,
                filename=tok_file.name,
                local_dir=str(tok_file.parent),
            )

        self._tok = Tokenizer.from_file(str(tok_file))
        self._special_to_id = {}
        for token in self._SPECIALS:
            token_id = self._tok.token_to_id(token)
            if token_id is not None:
                self._special_to_id[token] = token_id

        self.pad_token_id = self._special_to_id["<|endoftext|>"]
        self.eos_token_id = self.pad_token_id

        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def encode(self, text, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        text = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if self.add_thinking:
                if self.thinking_template == "tagged":
                    text += "<think>\n"
            else:
                text += "<think>\n\n</think>\n\n"
        return text


def text_to_token_ids(text, tokenizer, chat_wrapped=None):
    encoded = tokenizer.encode(text, chat_wrapped=chat_wrapped)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    if isinstance(token_ids, torch.Tensor):
        flat = token_ids.squeeze(0).tolist()
    else:
        flat = token_ids
    return tokenizer.decode(flat)


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for batch_index, (input_batch, target_batch) in enumerate(data_loader):
        if batch_index >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer, chat_wrapped=False).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=int(model.cfg["context_length"]),
            use_cache=True,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    use_amp=False,
    gradient_accumulation_steps=1,
    compile_model=False,
    eval_callback=None,
):
    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    forward_model = model
    if compile_model:
        try:
            forward_model = torch.compile(model)
            print("[OPT] torch.compile enabled")
        except Exception as exc:
            print(f"[OPT] torch.compile unavailable ({exc}), continuing without it")
            forward_model = model

    if amp_enabled:
        print(f"[OPT] AMP enabled (dtype={amp_dtype}, grad_scaling={scaler_enabled})")
    if gradient_accumulation_steps > 1:
        print(f"[OPT] Gradient accumulation: {gradient_accumulation_steps} steps (effective batch = {train_loader.batch_size * gradient_accumulation_steps})")

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for step_in_epoch, (input_batch, target_batch) in enumerate(train_loader):
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = calc_loss_batch(input_batch, target_batch, forward_model, device)
                loss = loss / gradient_accumulation_steps

            if scaler_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                if scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
                if eval_callback is not None:
                    eval_callback(
                        epoch=epoch + 1,
                        step=global_step,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        tokens_seen=tokens_seen,
                        train_losses=train_losses,
                        val_losses=val_losses,
                        track_tokens_seen=track_tokens_seen,
                        model=model,
                        optimizer=optimizer,
                    )

        # flush remaining accumulated gradients at end of epoch
        if (step_in_epoch + 1) % gradient_accumulation_steps != 0:
            if scaler_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        generate_and_print_sample(
            model,
            tokenizer,
            device,
            start_context,
        )

    return train_losses, val_losses, track_tokens_seen
