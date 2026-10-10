# Local AI field notes

I'm learning to run AI on my own hardware, and I'm writing down what I find as I go.

I believe it matters whether people rent AI or own it. Open-weight models and open source AI are how people get to own it, and open innovation is how it keeps getting better. When a model runs on your machine, you can look inside it and change it, and it'll still be there next year, whatever the company that made it decides.

I'm not an engineer. These notes are for people like me who want to understand how local AI works and who builds it.

## What's in here

- [people.md](people.md) lists the builders behind local AI, sorted by the part of the stack they work on, with handles you can follow. They're also on my X list, [Local AI Builders](https://x.com/i/lists/2108979978985742481)
- [notes/](notes/) has dated logs of what I tried, what worked and what broke
- [models.md](models.md) tracks every model I've run, with how much memory it used and how fast it went
- [glossary.md](glossary.md) explains the terms that confused me when I started
- [data/people.csv](data/people.csv) has the people list in a format you can sort or import

## My setup

Mac mini, Apple M5 Pro, 64 GB unified memory.

## Run the same test on your Mac

`bash scripts/bench.sh` installs mlx-lm in a separate environment, picks models that fit your memory and logs speed and memory use. Pause anything else using the GPU first. Results land in `results/` and `models.md`.

## Add someone

If a person belongs on the list and isn't there, open a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for the two rules.

## About

I'm Pratik ([@pratikg](https://x.com/pratikg) on X, [pratikgx](https://huggingface.co/pratikgx) on Hugging Face). I work at Eigen Labs. When I'm not using my Mac mini, it serves open-weight models on [Darkbloom](https://x.com/darkbloomai), an open inference network that runs on idle Macs. This repo is personal.
