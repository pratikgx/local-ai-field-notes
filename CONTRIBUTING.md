# Contributing

This repo is me learning local AI in public. Three ways to help:

## Share your results

Run the same test on your Mac and share the numbers, so we can see how other chips and memory sizes compare.

1. You need a Mac with Apple silicon. If `python3` isn't installed, macOS will offer to install Apple's Command Line Tools. Accept, then run the script again.
2. Pause anything else that uses the GPU, then run `bash scripts/bench.sh` from the repo folder.
3. [Open an issue](https://github.com/pratikgx/local-ai-field-notes/issues/new) titled with your chip and memory, for example "M4 Max, 128 GB". Paste the contents of the new file in `results/`. Add anything that broke and how you fixed it.

The results file includes your chip, memory, macOS version and Mac model, and nothing else about you. The script also edits your copy of `models.md`, which is my log, so please don't include that change in a pull request.

## Add someone

Two rules keep the people list useful.

1. **Handles need proof.** Add an X handle only if the person shows it on their own GitHub profile or website. Put that link in your pull request description.
2. **One true line.** Say what the person actually built or does, in plain words. "Wrote llama.cpp" works. "Visionary AI leader" doesn't.

Add the person to both `people.md` and `data/people.csv`, in the section that matches their work.

## Fix something

Corrections are welcome. If a number, a handle or an explanation is wrong, open a pull request or an issue.
