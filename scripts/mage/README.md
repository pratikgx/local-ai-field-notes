# Mage-VL-Cowork bench scripts

Scripts for the Oct 30 test of Microsoft's [Mage-VL-Cowork](https://huggingface.co/microsoft/Mage-VL-Cowork) 4B on a Mac with Apple silicon. The write-up is in [notes/2026-10-30-mage-vl-cowork.md](../../notes/2026-10-30-mage-vl-cowork.md).

## What's here

| File | What it does |
| --- | --- |
| `remap.py` | Renames Microsoft's weight keys to the `mage_vl` layout mlx-vlm 0.7.11 reads. Tensors are copied byte for byte. |
| `common.py` | Loading, prompt formatting and timed generation shared by the other scripts. |
| `filler_text.py` | The filler text and context lengths for the text-speed test. |
| `mage_bench.py` | Text speed at 512, 4K, 16K and 32K, plus one agent move from a screenshot at 1280x800 and 1080p. |
| `demo.py` | The 3 demo tasks, in a throwaway sandbox folder. |
| `weight_bytes.py` | Counts the bytes in each part of a build, for the bandwidth estimates. |
| `summarize.py` | Builds `results/2026-10-30-mage-vl-cowork.json` and `.csv` from the raw files. |
| `make_chart.py`, `make_explained_chart.py` | Draw the two charts in `notes/img/`. |
| `system_prompt.txt` | Microsoft's system prompt for the model, unchanged. |
| `screens/` | The mock Settings, spreadsheet and file manager screens (HTML and PNG at 1280x800 and 1080p), plus `sales.csv`. |

## Run it

Pause anything else that uses the GPU first. From the repo root:

```bash
python3 -m venv .venv-mage && source .venv-mage/bin/activate
pip install mlx==0.32.4 mlx-vlm==0.7.11 huggingface_hub pillow matplotlib

# 1. Download Microsoft's weights and remap the keys
hf download microsoft/Mage-VL-Cowork --local-dir converted/mage-hf
python scripts/mage/remap.py converted/mage-hf converted/mage-remapped

# 2. Make the two MLX builds
python -m mlx_vlm.convert --hf-path converted/mage-remapped --mlx-path converted/mlx_bf16
python -m mlx_vlm.convert --hf-path converted/mage-remapped --mlx-path converted/mlx_4bit -q --q-bits 4 --q-group-size 64

# 3. Speed and one agent move, then the 3 demo tasks, for each build
python scripts/mage/mage_bench.py converted/mlx_4bit mlx-4bit-g64 converted/runs/4bit
python scripts/mage/demo.py converted/mlx_4bit mlx-4bit-g64 converted/runs/4bit
python scripts/mage/mage_bench.py converted/mlx_bf16 mlx-bf16 converted/runs/bf16
python scripts/mage/demo.py converted/mlx_bf16 mlx-bf16 converted/runs/bf16
```

Or skip steps 1 and 2 and use the 4-bit build I uploaded: `hf download pratikgx/Mage-VL-Cowork-MLX-4bit --local-dir converted/mlx_4bit`.

`converted/` is already in `.gitignore`. Each run writes `bench.json` and `demo.json` to the folder you name.

To rebuild the summary and charts from the raw files in `results/2026-10-30-mage-vl-cowork/` (no Mac needed):

```bash
python scripts/mage/summarize.py
python scripts/mage/make_chart.py
python scripts/mage/make_explained_chart.py
```

## Notes

- **The 4-bit build keeps the vision part at full precision.** Only the language model is 4-bit, which works out to 5.3 bits per weight and 3.1 GB. The exact convert flags from Oct 30 weren't saved. After converting, check that `config.json` shows `bits: 4` and `group_size: 64`, and that the file is about 3.1 GB.
- **The filler text differs.** `filler_text.py` is a stand-in for an earlier, unpublished helper. The context lengths are the same, but the words aren't, so expect speeds close to mine rather than identical.
- **Scratch files.** The scripts save JPEG copies of the screens next to the PNGs (`screens/*_q95.jpg`). You can delete them.
- **Clicks aren't executed.** `demo.py` records each click and scores it against the target. Screenshot and wait calls just get the same screenshot back. Shell commands run only inside a temporary sandbox folder, after a filter that blocks sudo, curl, wget, ssh, rm -rf and paths outside the sandbox.
- **Results.** Send yours as an issue, as described in [CONTRIBUTING.md](../../CONTRIBUTING.md).
