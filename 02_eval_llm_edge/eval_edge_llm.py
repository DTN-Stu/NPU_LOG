"""
Edge-LLM Algorithmic Accuracy & Ablation Suite for SNAX FP4 Subsystem (Tier A).

Evaluates small/edge LLMs (e.g., Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B/3B, SmolLM2-1.7B):
  1. Perplexity (PPL) on WikiText-2 and C4
  2. Zero-shot Accuracy on PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, MMLU (via lm-eval)
  3. Fair Iso-Bitwidth Ablation (4.50 b/elem at B=16):
       FP16 vs. MXFP4 (B=32) vs. MXFP4_g16 (floor & ceil) vs. NVFP4 vs. Log8-FP4 vs. Full RTL (CSD + Exp-Free + V-Transpose)
  4. Attention Output (O) L1 Error & Cosine Similarity: Standard FP16 Softmax vs. Full RTL Exp-Free Softmax
"""

import argparse
import contextlib
import csv
import gc
import math
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantizers import (
    exp_free_softmax_requantizer,
    quantize_log8_fp4,
    quantize_log8_fp4_csd,
    quantize_log8_fp4_exact,
    quantize_log8_fp4_exact_csd,
    quantize_mxfp4,
    quantize_mxfp4_g16_ceil,
    quantize_mxfp4_g16_floor,
    quantize_mxfp4_g32_floor,
    quantize_nvfp4,
)

QUANT_FNS: Dict[str, Optional[Callable[[torch.Tensor], torch.Tensor]]] = {
    "fp16": None,
    "mxfp4": quantize_mxfp4,
    "mxfp4_g32_floor": quantize_mxfp4_g32_floor,
    "mxfp4_g16_floor": quantize_mxfp4_g16_floor,
    "mxfp4_g16_ceil": quantize_mxfp4_g16_ceil,
    "nvfp4": quantize_nvfp4,
    "log8_fp4": quantize_log8_fp4,
    "log8_fp4_csd": quantize_log8_fp4_csd,
    "log8_fp4_exact": quantize_log8_fp4_exact,
    "log8_fp4_exact_csd": quantize_log8_fp4_exact_csd,
}

# Mapping from config name -> (weight_mode, act_mode, kv_mode, use_exp_free_softmax)
CONFIG_SPECS: Dict[str, tuple] = {
    "fp16": ("fp16", "fp16", "fp16", False),
    "mxfp4": ("mxfp4", "mxfp4", "mxfp4", False),
    "mxfp4_g32_floor": ("mxfp4_g32_floor", "mxfp4_g32_floor", "mxfp4_g32_floor", False),
    "mxfp4_g16_floor": ("mxfp4_g16_floor", "mxfp4_g16_floor", "mxfp4_g16_floor", False),
    "mxfp4_g16_ceil": ("mxfp4_g16_ceil", "mxfp4_g16_ceil", "mxfp4_g16_ceil", False),
    "nvfp4": ("nvfp4", "nvfp4", "nvfp4", False),
    "log8_fp4": ("log8_fp4", "log8_fp4", "log8_fp4", False),
    "log8_fp4_csd": ("log8_fp4_csd", "log8_fp4_csd", "log8_fp4_csd", False),
    "log8_fp4_expfree": ("log8_fp4_csd", "log8_fp4_csd", "log8_fp4_csd", True),
    "log8_fp4_exact": ("log8_fp4_exact", "log8_fp4_exact", "log8_fp4_exact", False),
    "log8_fp4_exactw_csd": ("log8_fp4_exact_csd", "log8_fp4_csd", "log8_fp4_csd", False),
    "log8_fp4_exactw_expfree": ("log8_fp4_exact_csd", "log8_fp4_csd", "log8_fp4_csd", True),
}


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve HF token from CLI, environment variables, or Google Colab Secrets."""
    token = (
        cli_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        try:
            from google.colab import userdata  # type: ignore

            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None

    if isinstance(token, str):
        token = token.strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        return token
    return None


def _parse_major_minor(version_str: str) -> tuple:
    match = re.match(r"^\s*(\d+)\.(\d+)", version_str)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def load_causal_lm(
    model_id: str,
    device: str,
    hf_token: Optional[str] = None,
) -> nn.Module:
    """Load HF CausalLM in FP16 with SDPA enabled across Transformers 4.x and 5.x."""
    dtype_key = (
        "dtype"
        if _parse_major_minor(getattr(transformers, "__version__", "0.0")) >= (4, 56)
        else "torch_dtype"
    )
    load_kwargs = {
        dtype_key: torch.float16,
        "attn_implementation": "sdpa",
        "token": hf_token,
    }
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    return model.to(device).eval()


def load_wikitext2_ids(
    tokenizer,
    hf_token: Optional[str] = None,
) -> torch.Tensor:
    """Load and tokenize WikiText-2 test split using canonical namespace."""
    candidates = [
        ("Salesforce/wikitext", "wikitext-2-raw-v1"),
        ("wikitext", "wikitext-2-raw-v1"),
    ]
    errors = []
    for repo_id, config_name in candidates:
        try:
            ds = load_dataset(repo_id, config_name, split="test", token=hf_token)
            text = "\n\n".join(ds["text"])
            return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        except Exception as exc:
            errors.append(f"{repo_id}/{config_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Failed to load WikiText-2 test split:\n" + "\n".join(errors))


def load_c4_ids(
    tokenizer,
    seq_len: int,
    max_chunks: int,
    hf_token: Optional[str] = None,
) -> torch.Tensor:
    """Stream single C4 validation shard cleanly across datasets v3.x and v4.x."""
    shard_rel = "en/c4-validation.00000-of-00008.json.gz"
    shard_url = f"https://huggingface.co/datasets/allenai/c4/resolve/main/{shard_rel}"
    attempts = [
        (
            "allenai/c4 (data_files)",
            lambda: load_dataset(
                "allenai/c4",
                data_files={"validation": shard_rel},
                split="validation",
                streaming=True,
                token=hf_token,
            ),
        ),
        (
            "json direct shard URL",
            lambda: load_dataset(
                "json",
                data_files={"validation": shard_url},
                split="validation",
                streaming=True,
                token=hf_token,
            ),
        ),
        (
            "allenai/c4 (en)",
            lambda: load_dataset(
                "allenai/c4",
                "en",
                split="validation",
                streaming=True,
                token=hf_token,
            ),
        ),
    ]

    stream_iter = None
    first_row = None
    errors = []
    for label, loader_fn in attempts:
        try:
            ds = loader_fn()
            it = iter(ds)
            first_row = next(it)
            stream_iter = it
            break
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    if stream_iter is None or first_row is None:
        raise RuntimeError("Failed to stream allenai/c4 validation split:\n" + "\n".join(errors))

    target_tokens = min(max_chunks, 128) * seq_len
    sep_ids = tokenizer("\n\n", add_special_tokens=False).input_ids
    all_ids: List[int] = []

    def _rows():
        yield first_row
        yield from stream_iter

    for doc_idx, row in enumerate(_rows()):
        text = row.get("text", "")
        if text:
            if all_ids and sep_ids:
                all_ids.extend(sep_ids)
            all_ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
        if len(all_ids) >= target_tokens or doc_idx + 1 >= 2048:
            break

    if not all_ids:
        raise RuntimeError("C4 stream returned zero tokens.")
    return torch.tensor(all_ids, dtype=torch.long).unsqueeze(0)


class QuantLinearWrapper(nn.Module):
    """Wraps nn.Linear to simulate W4A4 block-scaled quantization on the fly."""

    def __init__(
        self,
        linear: nn.Linear,
        quant_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        weight_quant_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.linear = linear
        self.quant_fn = quant_fn
        w_fn = quant_fn if weight_quant_fn is None else weight_quant_fn
        # Pre-quantize static weights once for speed
        if w_fn is not None:
            with torch.no_grad():
                q_w = w_fn(linear.weight.data).detach()
        else:
            q_w = linear.weight.data.detach()
        self.register_buffer("q_weight", q_w, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_fn is not None:
            x = self.quant_fn(x)
        q_weight = self.q_weight
        if q_weight.dtype != x.dtype or q_weight.device != x.device:
            q_weight = q_weight.to(device=x.device, dtype=x.dtype)
        bias = self.linear.bias
        if bias is not None and (bias.dtype != x.dtype or bias.device != x.device):
            bias = bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, q_weight, bias)


def apply_linear_quantization(
    model: nn.Module,
    quant_mode: str,
    quant_lm_head: bool = False,
    weight_quant_mode: Optional[str] = None,
) -> List[tuple]:
    """Replace nn.Linear modules with QuantLinearWrapper. Returns list for restoration."""
    act_fn = QUANT_FNS[quant_mode]
    w_fn = QUANT_FNS[weight_quant_mode] if weight_quant_mode is not None else act_fn
    if act_fn is None and w_fn is None:
        return []

    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if not quant_lm_head and "lm_head" in name:
                continue
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            wrapped = QuantLinearWrapper(module, act_fn, weight_quant_fn=w_fn)
            setattr(parent, child_name, wrapped)
            replaced.append((parent, child_name, module))
    return replaced


def restore_linear_modules(replaced: List[tuple]) -> None:
    for parent, child_name, orig_module in replaced:
        setattr(parent, child_name, orig_module)


@contextlib.contextmanager
def patch_attention_sdpa(
    use_exp_free: bool,
    kv_quant_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    base_scale_mode: str = "log2e",
    v_transpose: bool = True,
    metrics_log: Optional[List[Dict[str, float]]] = None,
):
    """
    Context manager that intercepts F.scaled_dot_product_attention (used by
    Llama, Qwen2, SmolLM2, Mistral in eager/sdpa mode) to simulate:
      - Q/K quantization along head_dim and V quantization along token axis N_KV (v_transpose=True,
        matching snax_transpose_buffer.sv) or channel axis D_V (v_transpose=False)
      - Exp-free Log8-FP4 Softmax Requantizer
      - Optional L1 / Cosine Similarity measurement of O vs FP16 Softmax
    """
    orig_sdpa = F.scaled_dot_product_attention

    def custom_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
        **_extra_kwargs,
    ) -> torch.Tensor:
        if query.shape[1] != key.shape[1]:
            repeat_factor = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        q_in = kv_quant_fn(query) if kv_quant_fn is not None else query
        k_in = kv_quant_fn(key) if kv_quant_fn is not None else key
        if kv_quant_fn is not None:
            if v_transpose:
                # Match snax_transpose_buffer.sv: quantize V across blocks of 16 tokens along N_KV
                v_in = kv_quant_fn(value.transpose(-2, -1).contiguous()).transpose(-2, -1)
            else:
                v_in = kv_quant_fn(value)
        else:
            v_in = value

        d_k = q_in.shape[-1]
        scale_factor = (1.0 / math.sqrt(d_k)) if scale is None else scale
        scores = torch.matmul(q_in.float(), k_in.float().transpose(-2, -1)) * scale_factor

        q_len, k_len = scores.shape[-2], scores.shape[-1]
        mask_bool = None
        if is_causal:
            causal = torch.ones((q_len, k_len), dtype=torch.bool, device=scores.device).tril(
                diagonal=k_len - q_len
            )
            mask_bool = causal
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                m = attn_mask.to(device=scores.device)
            else:
                m = (attn_mask > -1e4).to(device=scores.device)
            mask_bool = m if mask_bool is None else (mask_bool & m)

        # Reference FP16 Softmax output (when logging attention error)
        if metrics_log is not None:
            ref_scores = scores.masked_fill(~mask_bool, float("-inf")) if mask_bool is not None else scores
            p_ref = torch.nan_to_num(torch.softmax(ref_scores, dim=-1), nan=0.0)
            o_ref = torch.matmul(p_ref, value.float())

        if use_exp_free:
            p_hat = exp_free_softmax_requantizer(scores, attn_mask=mask_bool, base_scale_mode=base_scale_mode)
            o_out = torch.matmul(p_hat.float(), v_in.float()).to(query.dtype)
        else:
            if mask_bool is not None:
                scores = scores.masked_fill(~mask_bool, float("-inf"))
            p_std = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            o_out = torch.matmul(p_std, v_in.float()).to(query.dtype)

        if metrics_log is not None:
            o_f = o_out.float()
            d_v = o_f.shape[-1]
            l1_err = (o_f - o_ref).abs().mean().item()
            rel_l1 = ((o_f - o_ref).abs().sum() / (o_ref.abs().sum() + 1e-12)).item()
            cos_sim = F.cosine_similarity(o_f.reshape(-1, d_v), o_ref.reshape(-1, d_v), dim=-1).mean().item()
            metrics_log.append({"l1_err": l1_err, "rel_l1": rel_l1, "cos_sim": cos_sim})

        return o_out

    F.scaled_dot_product_attention = custom_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig_sdpa


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    enc: torch.Tensor,
    dataset_name: str,
    seq_len: int = 1024,
    max_chunks: int = 64,
    device: str = "cuda",
) -> float:
    """Sliding-window / chunked NLL perplexity on pre-tokenized dataset IDs."""
    total_tokens = enc.shape[1]
    num_chunks = min(max_chunks, total_tokens // seq_len)
    if num_chunks == 0:
        raise ValueError(
            f"Dataset '{dataset_name}' only has {total_tokens} tokens, fewer than seq_len={seq_len}."
        )

    nlls = []
    for i in tqdm(range(num_chunks), desc=f"PPL ({dataset_name})", leave=False):
        batch = enc[:, i * seq_len : (i + 1) * seq_len].to(device)
        outputs = model(batch, labels=batch, use_cache=False)
        nlls.append(outputs.loss.float().detach().cpu())

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl


@torch.no_grad()
def evaluate_attention_error(
    model: nn.Module,
    enc: torch.Tensor,
    seq_len: int = 512,
    num_batches: int = 4,
    base_scale_mode: str = "log2e",
    v_transpose: bool = True,
    use_csd: bool = True,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure L1 error & Cosine Similarity of Attention O vs FP16 Softmax."""
    metrics_log: List[Dict[str, float]] = []
    kv_fn = quantize_log8_fp4_csd if use_csd else quantize_log8_fp4
    total_batches = min(num_batches, enc.shape[1] // seq_len)
    if total_batches == 0:
        print(f"[WARN] Tokenized text ({enc.shape[1]} tokens) shorter than seq_len={seq_len} for attention error.")
        return {"l1_err": 0.0, "rel_l1": 0.0, "cos_sim": 1.0}

    with patch_attention_sdpa(
        use_exp_free=True,
        kv_quant_fn=kv_fn,
        base_scale_mode=base_scale_mode,
        v_transpose=v_transpose,
        metrics_log=metrics_log,
    ):
        for i in range(total_batches):
            batch = enc[:, i * seq_len : (i + 1) * seq_len].to(device)
            model(batch, use_cache=False)

    if not metrics_log:
        print("[WARN] SDPA hook was not triggered during attention error evaluation.")
        return {"l1_err": 0.0, "rel_l1": 0.0, "cos_sim": 1.0}

    return {
        "l1_err": sum(m["l1_err"] for m in metrics_log) / len(metrics_log),
        "rel_l1": sum(m["rel_l1"] for m in metrics_log) / len(metrics_log),
        "cos_sim": sum(m["cos_sim"] for m in metrics_log) / len(metrics_log),
    }


def evaluate_zeroshot(
    model: nn.Module,
    tokenizer,
    tasks: List[str],
    batch_size: int = 8,
    limit: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """Run lm-evaluation-harness on the active (possibly quantized) model."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("[WARN] lm-eval not installed. Run `pip install lm-eval` to enable zero-shot tasks.")
        return {}

    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        limit=limit,
        log_samples=False,
    )
    scores = {}
    for task_name, task_res in results["results"].items():
        acc = task_res.get("acc_norm,none", task_res.get("acc,none", None))
        if acc is not None:
            scores[task_name] = acc * 100.0

    del lm_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def save_results_csv(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    """Write accumulated evaluation rows with unioned ordered fieldnames."""
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Tier-A Edge LLM Evaluation for SNAX FP4 Subsystem")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="HF model ID or local path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional HuggingFace token (falls back to HF_TOKEN env var or Google Colab Secrets)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "fp16",
            "mxfp4",
            "mxfp4_g16_floor",
            "mxfp4_g16_ceil",
            "nvfp4",
            "log8_fp4",
            "log8_fp4_expfree",
            "log8_fp4_exactw_expfree",
        ],
        choices=list(CONFIG_SPECS.keys()),
        help="Quantization & Softmax configurations to benchmark",
    )
    parser.add_argument("--eval_ppl", action="store_true", help="Evaluate WikiText-2 and C4 Perplexity")
    parser.add_argument("--eval_attn_err", action="store_true", help="Measure Attention O L1 & Cosine Sim")
    parser.add_argument("--eval_zeroshot", action="store_true", help="Evaluate Zero-shot tasks via lm-eval")
    parser.add_argument(
        "--zeroshot_tasks",
        nargs="+",
        default=["piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "mmlu"],
    )
    parser.add_argument("--zeroshot_batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_chunks", type=int, default=64)
    parser.add_argument("--zeroshot_limit", type=int, default=None, help="Optional sample limit per task for quick test")
    parser.add_argument("--base_scale_mode", type=str, default="log2e", choices=["log2e", "rtl_ln2"])
    parser.add_argument(
        "--v_quant_axis",
        type=str,
        default="token",
        choices=["token", "channel"],
        help="V quantization axis in attention: 'token' (N_KV, matches snax_transpose_buffer.sv) or 'channel' (D_V)",
    )
    parser.add_argument("--quant_lm_head", action="store_true", help="Also quantize lm_head")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    if not (args.eval_ppl or args.eval_attn_err or args.eval_zeroshot):
        args.eval_ppl = True
        args.eval_attn_err = True

    v_transpose = args.v_quant_axis == "token"
    hf_token = resolve_hf_token(args.hf_token)
    if hf_token:
        print(f"[Auth] HF_TOKEN configured (len={len(hf_token)}).")
    else:
        print("[Auth] HF_TOKEN not set; using public anonymous HuggingFace access.")

    os.makedirs(args.out_dir, exist_ok=True)
    model_tag = args.model.replace("/", "_")
    csv_path = Path(args.out_dir) / f"eval_{model_tag}.csv"

    print(f"Loading {args.model} on {args.device} (attn_implementation='sdpa', V-axis={args.v_quant_axis})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_lm(args.model, args.device, hf_token=hf_token)

    enc_wt2 = None
    enc_c4 = None
    if args.eval_ppl or args.eval_attn_err:
        print("Pre-loading & tokenizing WikiText-2 test split...")
        enc_wt2 = load_wikitext2_ids(tokenizer, hf_token=hf_token)
        print(f"  WikiText-2 tokens: {enc_wt2.shape[1]}")
    if args.eval_ppl:
        print("Pre-loading & tokenizing C4 validation stream...")
        enc_c4 = load_c4_ids(tokenizer, args.seq_len, args.max_chunks, hf_token=hf_token)
        print(f"  C4 tokens: {enc_c4.shape[1]}")

    all_rows = []

    if args.eval_attn_err and enc_wt2 is not None:
        attn_stats = evaluate_attention_error(
            model,
            enc_wt2,
            seq_len=min(512, args.seq_len),
            base_scale_mode=args.base_scale_mode,
            v_transpose=v_transpose,
            use_csd=True,
            device=args.device,
        )
        print(
            f"\n[Attention O Error | Full RTL Exp-Free Log8-FP4 CSD Softmax ({args.base_scale_mode}, V={args.v_quant_axis}) vs FP16] "
            f"L1={attn_stats['l1_err']:.6f}, Rel-L1={attn_stats['rel_l1']*100:.2f}%, "
            f"CosSim={attn_stats['cos_sim']:.6f}\n"
        )

    for cfg in args.configs:
        w_mode, a_mode, kv_mode, use_exp_free = CONFIG_SPECS[cfg]
        print(f"=== Evaluating config: {cfg} (W={w_mode}, A={a_mode}, KV={kv_mode}, exp_free={use_exp_free}) ===")
        kv_fn = QUANT_FNS[kv_mode]

        replaced = apply_linear_quantization(
            model,
            a_mode,
            quant_lm_head=args.quant_lm_head,
            weight_quant_mode=w_mode,
        )
        row = {"model": args.model, "config": cfg}

        with patch_attention_sdpa(
            use_exp_free=use_exp_free,
            kv_quant_fn=kv_fn,
            base_scale_mode=args.base_scale_mode,
            v_transpose=v_transpose,
        ):
            if args.eval_ppl:
                row["ppl_wikitext2"] = round(
                    evaluate_perplexity(model, enc_wt2, "wikitext2", args.seq_len, args.max_chunks, args.device), 3
                )
                row["ppl_c4"] = round(
                    evaluate_perplexity(model, enc_c4, "c4", args.seq_len, args.max_chunks, args.device), 3
                )
            if args.eval_zeroshot:
                zs = evaluate_zeroshot(
                    model,
                    tokenizer,
                    args.zeroshot_tasks,
                    batch_size=args.zeroshot_batch_size,
                    limit=args.zeroshot_limit,
                    device=args.device,
                )
                for k, v in zs.items():
                    row[f"zs_{k}"] = round(v, 2)

        restore_linear_modules(replaced)
        replaced.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_rows.append(row)
        save_results_csv(csv_path, all_rows)
        print(f"Result [{cfg}]: {row} (saved -> {csv_path})")

    if all_rows:
        print(f"\nCompleted! Summary CSV at: {csv_path}")


if __name__ == "__main__":
    main()
