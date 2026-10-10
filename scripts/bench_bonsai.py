"""
Benchmark PrismML's Ternary Bonsai 2 27B MLX pack, using the same prompt and
numbers as scripts/bench.py.

Plain mlx-lm can't run this pack. Its ternary weights are stored in a rotated
(Hadamard) basis, so each layer has to rotate its input the same way before
multiplying. This script does that rotation itself, on top of MLX's standard
Qwen3.5 model code.

Setup (needs Python 3.11 or newer):
    python3 -m venv .venv-bonsai
    .venv-bonsai/bin/pip install "mlx==0.32.0" "mlx-lm==0.31.3" tokenizers jinja2
    hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --local-dir bonsai-pack

Run:
    .venv-bonsai/bin/python scripts/bench_bonsai.py bonsai-pack
"""
import json, math, sys, time
from pathlib import Path

import mlx.core as mx
from mlx import nn
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

PROMPT = "Explain in three short paragraphs why someone might run AI models on their own computer."
MAX_TOKENS = 256


class Rotated(nn.Module):
    """A 2-bit layer whose weights live in a rotated basis.

    Normal layer: y = W' (H D x), where D flips signs and H is a normalized
    Hadamard transform in blocks. Embedding: rotate the looked-up row back,
    out = D H e'.
    """

    def __init__(self, w, scales, biases, signs, block, embedding):
        super().__init__()
        self.w, self.scales, self.biases, self.signs = w, scales, biases, signs
        self.block, self.embedding = block, embedding

    def hadamard(self, x):
        shape = x.shape
        x = mx.hadamard_transform(x.reshape(-1, self.block), scale=1 / math.sqrt(self.block))
        return x.reshape(shape)

    def __call__(self, x):
        if self.embedding:
            idx = x.reshape(-1)
            e = mx.dequantize(self.w[idx], self.scales[idx], self.biases[idx], group_size=128, bits=2)
            e = e.astype(mx.float32).reshape(*x.shape, -1)
            return (self.hadamard(e) * self.signs).astype(mx.float16)
        x = self.hadamard(x.astype(mx.float32) * self.signs).astype(mx.float16)
        return mx.quantized_matmul(x, self.w, self.scales, self.biases,
                                   transpose=True, group_size=128, bits=2)


def load(pack):
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    cfg = json.loads((pack / "config.json").read_text())
    model = TextModel(TextModelArgs.from_dict(cfg["text_config"]))
    weights = mx.load(str(pack / "model.safetensors"))
    rotated = {m["path"]: m for m in cfg["modules"]}
    for path, m in rotated.items():
        k = "language_model." + path
        parent = model
        parts = path.split(".")
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        setattr(parent, parts[-1], Rotated(weights[k + ".weight"], weights[k + ".scales"],
                                           weights[k + ".biases"], weights[k + ".signs"],
                                           m["block"], m["embedding"]))
    plain = {k[len("language_model."):]: v for k, v in weights.items()
             if k.startswith("language_model.")
             and k.rsplit(".", 1)[0][len("language_model."):] not in rotated}
    model.load_weights(list(plain.items()), strict=False)
    model.eval()
    mx.eval(model.parameters())
    return model


def main():
    pack = Path(sys.argv[1])
    model = load(pack)
    tok = Tokenizer.from_file(str(pack / "tokenizer.json"))
    template = (pack / "chat_template.jinja").read_text()
    text = (ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            .from_string(template)
            .render(messages=[{"role": "user", "content": PROMPT}], add_generation_prompt=True))
    ids = tok.encode(text, add_special_tokens=False).ids
    eos = json.loads((pack / "generation_config.json").read_text()).get("eos_token_id")
    stop = set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    stop |= {i for i in (tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")) if i is not None}

    mx.reset_peak_memory()
    cache = model.make_cache()
    t0 = time.perf_counter()
    y = mx.argmax(model(mx.array([ids]), cache=cache)[:, -1, :], axis=-1)
    mx.eval(y)
    t1 = time.perf_counter()
    out = [y.item()]
    while len(out) < MAX_TOKENS and out[-1] not in stop:
        y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(y)
        out.append(y.item())
    t2 = time.perf_counter()

    print("==========")
    print(tok.decode([t for t in out if t not in stop]))
    print("==========")
    print(f"Prompt: {len(ids)} tokens, {len(ids) / (t1 - t0):.3f} tokens-per-sec")
    print(f"Generation: {len(out) - (out[-1] in stop)} tokens, {(len(out) - 1) / (t2 - t1):.3f} tokens-per-sec")
    print(f"Peak memory: {mx.get_peak_memory() / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
