"""Filler text for the text-speed test in mage_bench.py.

The prompt is a short instruction (PERF_HEAD) followed by random sentences built
from FILLER_WORDS, trimmed to each target context length. Only the token count
matters for speed, not the words.

Note: the Oct 30 run used an earlier, unpublished copy of this helper with its
own word list and head text. Prompt lengths match (512, 4K, 16K, 32K tokens),
but the exact words differ, so your speeds should be close but not identical.
"""
FILLER_SEED = 1234
PERF_CONTEXTS = [512, 4096, 16384, 32768]
PERF_HEAD = "Read the notes below, then write a short summary of them.\n\n"
FILLER_WORDS = (
    "the a an of to and in on for with from by at as is was are were be been "
    "machine memory model token screen window file folder desk light river stone "
    "garden city market table window paper morning evening quiet simple careful "
    "small large early late open closed bright dark warm cold fast slow steady "
    "build run test read write check move click type wait start finish keep share"
).split()
