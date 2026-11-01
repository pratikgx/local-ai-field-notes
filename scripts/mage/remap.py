# Remap microsoft/Mage-VL-Cowork (llava_onevision2 key layout) to the mage_vl layout mlx-vlm 0.7.11 understands.
# Only key prefixes and config model_type change; tensors are copied byte-for-byte (bf16).
import json, shutil, glob, os, sys
import mlx.core as mx
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    p = os.path.join(src, f)
    if os.path.isfile(p) and not f.endswith(".safetensors") and not f.endswith(".py") and f not in ("model.safetensors.index.json","config.json","preprocessor_config.json"):
        shutil.copy(p, dst)
cfg = json.load(open(os.path.join(src, "config.json")))
cfg.pop("auto_map", None)
cfg["model_type"] = "mage_vl"; cfg["architectures"] = ["MageVLForConditionalGeneration"]
cfg["vision_config"]["model_type"] = "mage_vl_vision"
cfg["eos_token_id"] = 151645; cfg["bos_token_id"] = 151643; cfg["tie_word_embeddings"] = False
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
pp = json.load(open(os.path.join(src, "preprocessor_config.json")))
pp["image_processor_type"] = "Qwen2VLImageProcessor"; pp.pop("processor_class", None)
json.dump(pp, open(os.path.join(dst, "preprocessor_config.json"), "w"), indent=2)
wm = {}
for sf in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
    w = mx.load(sf); out = {}
    for k, v in w.items():
        if k.startswith("visual."): k = "model." + k
        elif k.startswith("model.") and not k.startswith("model.language_model."): k = "model.language_model." + k[len("model."):]
        out[k] = v; wm[k] = os.path.basename(sf)
    mx.save_safetensors(os.path.join(dst, os.path.basename(sf)), out, metadata={"format": "pt"})
    print(sf, len(out))
json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
print("keys", len(wm))
