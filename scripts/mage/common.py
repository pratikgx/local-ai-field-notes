import time, json, os, mlx.core as mx
from mlx_vlm import load
from mlx_vlm.generate import stream_generate
HERE = os.path.dirname(os.path.abspath(__file__))
SYS = open(os.path.join(HERE, "system_prompt.txt")).read().rstrip("\n")
def render(processor, messages, enable_thinking=True):
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
def run(model, processor, prompt, images=None, max_tokens=256, temperature=0.0, seed=0, stop_at_eos=True, top_p=1.0):
    mx.random.seed(seed); mx.reset_peak_memory(); mx.clear_cache()
    t0 = time.perf_counter(); ttft = None; text = ""; last = None
    for r in stream_generate(model, processor, prompt, image=images, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed):
        if ttft is None: ttft = time.perf_counter() - t0
        text += r.text; last = r
    total = time.perf_counter() - t0
    return dict(text=text, ttft_s=ttft, total_s=total, prompt_tokens=last.prompt_tokens, gen_tokens=last.generation_tokens,
                prompt_tps=last.prompt_tps, gen_tps=last.generation_tps, peak_mem_gb=mx.get_peak_memory()/1e9, finish=last.finish_reason)
from PIL import Image
def jpeg95(png, out=None):
    out = out or png.rsplit(".",1)[0] + "_q95.jpg"
    Image.open(png).convert("RGB").save(out, format="JPEG", quality=95); return out
def ctx(instruction, previous="None", tool="computer_use"):
    lead = ("Please generate the next move according to the tool output, instruction and previous actions." if tool=="bash"
            else "Please generate the next move according to the UI screenshot, instruction and previous actions.")
    return f"{lead}\n\nInstruction: {instruction}\n\nPrevious actions:\n{previous}"
def first_msgs(instruction):
    return [{"role":"system","content":SYS},{"role":"user","content":[{"type":"image"},{"type":"text","text":"\n"+ctx(instruction)}]}]
