# Usage: hf_backlinks.py [--dry-run]
# Adds a short "Write-up on GitHub" section to the end of each Hugging Face card listed in
# data/huggingface.csv, linking the note(s) behind it. Rows without a note are skipped.
# The section sits between two hidden markers, so reruns only commit when the links change.
# Needs HF_TOKEN with write access and the huggingface_hub package, unless --dry-run.
import csv, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GITHUB = "https://github.com/pratikgx/local-ai-field-notes"
START, END = "<!-- github-field-notes:start -->", "<!-- github-field-notes:end -->"
URL_PREFIX = {"model": "", "dataset": "datasets/", "space": "spaces/"}

def section(notes):
    links = []
    for n in notes:
        with open(os.path.join(ROOT, n)) as f:
            title = f.readline().lstrip("# ").strip()
        links.append(f"[{title}]({GITHUB}/blob/main/{n})")
    joined = links[0] if len(links) == 1 else ", ".join(links[:-1]) + " and " + links[-1]
    return (f"{START}\n## Write-up on GitHub\n\n"
            f"How I ran this, what broke, and the scripts: {joined}. "
            f"It's part of [Local AI field notes]({GITHUB}), where I log every model I run on my Mac mini.\n{END}\n")

def with_section(card, block):
    if START in card and END in card:
        before, rest = card.split(START, 1)
        return before + block.rstrip("\n") + rest.split(END, 1)[1]
    return card.rstrip("\n") + "\n\n" + block

def main():
    dry = "--dry-run" in sys.argv
    api = None
    if not dry:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
    with open(os.path.join(ROOT, "data", "huggingface.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        notes = [n.strip() for n in r["note"].split(";") if n.strip()]
        if not notes or r["status"] != "public": continue
        url = f"https://huggingface.co/{URL_PREFIX[r['type']]}{r['repo']}/raw/main/README.md"
        card = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hf-backlinks"})).read().decode()
        new = with_section(card, section(notes))
        if new == card:
            print(f"up to date  {r['repo']}")
            continue
        if dry:
            print(f"would update {r['repo']}:\n{section(notes)}")
            continue
        api.upload_file(path_or_fileobj=new.encode(), path_in_repo="README.md", repo_id=r["repo"],
                        repo_type=r["type"], commit_message="Link the write-up on GitHub")
        print(f"updated     {r['repo']}")

if __name__ == "__main__":
    main()
