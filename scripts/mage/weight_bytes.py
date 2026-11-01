# Counts the bytes in each part of an MLX build (from the safetensors headers) to estimate memory traffic per decoded token.
# Usage: python3 weight_bytes.py <mlx_4bit_dir> <mlx_bf16_dir> > weight_bytes.json
import json, struct, glob, sys, os
def hdr(f):
    with open(f, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; return json.loads(fh.read(n))
out = {}
for d in sys.argv[1:]:
    g = {"vision": 0, "embed": 0, "lm_head": 0, "lm_layers_norms": 0}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        for k, v in hdr(f).items():
            if k == "__metadata__": continue
            b = v["data_offsets"][1] - v["data_offsets"][0]
            if k.startswith("vision_tower"): g["vision"] += b
            elif "embed_tokens" in k: g["embed"] += b
            elif "lm_head" in k: g["lm_head"] += b
            else: g["lm_layers_norms"] += b
    g["per_token_decode_bytes"] = g["lm_layers_norms"] + g["lm_head"]  # embedding is a row lookup, excluded; lm_head read fully
    out[os.path.basename(os.path.normpath(d))] = g
print(json.dumps(out, indent=1))
