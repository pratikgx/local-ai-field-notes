# Mini agent loop in Mage's native hybrid format. GUI actions are NOT executed (we only record them);
# computer_use 'screenshot'/'wait' get the same screenshot back. bash actions run only inside the sandbox after a safety check.
import sys, os, json, time, re, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
MP, LABEL, OUT = os.path.abspath(sys.argv[1]), sys.argv[2], os.path.abspath(sys.argv[3]); os.makedirs(OUT, exist_ok=True)
import tempfile
BASE = os.path.dirname(os.path.abspath(__file__))  # scripts/mage
os.chdir(BASE)  # screens/ paths are relative to this folder
SB = os.path.realpath(tempfile.mkdtemp(prefix="mage_sandbox_"))  # throwaway sandbox the model sees as /tmp_workspace
def reset_sandbox():
    shutil.rmtree(SB, ignore_errors=True); os.makedirs(SB)
    shutil.copy(os.path.join(BASE, "screens/sales.csv"), SB)
    for n, sz in [("Q3_budget.xlsx", 2048), ("report_final_v2.docx", 1024), ("team_photo.jpg", 4096), ("notes.txt", 64), ("invoice_0931.pdf", 512), ("roadmap.key", 256)]:
        open(os.path.join(SB, n), "wb").write(b"\0" * sz) if n != "notes.txt" else open(os.path.join(SB, n), "w").write("Ship Mage demo Friday.\n")
model, processor = load(MP)
TEMP = float(os.environ.get("DEMO_TEMP", "0")); TOP_P = float(os.environ.get("DEMO_TOP_P", "1.0"))
BAD = re.compile(r"\bsudo\b|\brm\s+-rf\s+[/~]|\bcurl\b|\bwget\b|\bssh\b|\bscp\b|\bosascript\b|\bopen\s|\bkill|\bshutdown|\breboot|\.\./|~/")
def safe(cmd):
    if BAD.search(cmd): return False
    for p in re.findall(r"(?<![\w.])(/[\w./-]+)", cmd):
        if p.startswith(SB) or p in ("/dev/null",) or p.startswith(("/usr/bin", "/bin")): continue
        return False
    return True
WS = "/tmp_workspace"  # model's native bash workspace (per model card); mapped onto the sandbox
def run_bash(cmd):
    cmd = cmd.replace(WS, SB)
    if not safe(cmd): return "[blocked by benchmark harness: command touches paths outside the sandbox]", False
    r = subprocess.run(["/bin/zsh", "-c", cmd], cwd=SB, capture_output=True, text=True, timeout=30,
                       env={"PATH": os.path.dirname(sys.executable) + ":/usr/bin:/bin", "HOME": SB})
    return (r.stdout + r.stderr).replace(SB, WS)[-4000:], True  # map real sandbox path back to the model's /tmp_workspace view
def parse(text):
    calls = []
    for m in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try: calls.append(json.loads(m))
        except Exception as e: calls.append({"parse_error": str(e), "raw": m})
    act = re.search(r"Action:\s*(.*)", text); return calls, (act.group(1).strip() if act else None)
TASKS = [
 {"id": "dark_mode", "image": "screens/settings_1080p.png", "instruction": "Turn on Dark Mode.",
  "bbox_px": [984*1.5, 215*1.5, 1030*1.5, 241*1.5], "kind": "gui"},
 {"id": "csv_total", "image": "screens/sheet_1080p.png", "instruction": "What is the total of the Revenue column in /tmp_workspace/sales.csv? Tell me the exact total.",
  "expected": "66026.61", "kind": "bash"},
 {"id": "rename_file", "image": "screens/finder_1080p.png", "instruction": "Rename report_final_v2.docx in /tmp_workspace to Q3_report.docx.",
  "kind": "either"},
]
results = []
for T in TASKS:
    reset_sandbox(); img = jpeg95(T["image"]); W, H = Image.open(img).size
    msgs = first_msgs(T["instruction"]); msgs[1]["content"][1]["text"] = "\n" + ctx(T["instruction"])
    images = [img]; steps = []; t_task = time.perf_counter()
    for step in range(1, 7):
        p = render(processor, msgs)
        r = run(model, processor, p, images=images if images else None, max_tokens=4096, temperature=TEMP, top_p=TOP_P)
        text = r["text"]; calls, act = parse(text)
        rec = {"step": step, "action_line": act, "tool_calls": calls, "secs": round(r["total_s"],2), "ttft_s": round(r["ttft_s"],2),
               "prompt_tokens": r["prompt_tokens"], "gen_tokens": r["gen_tokens"], "gen_tps": round(r["gen_tps"],1), "raw": text}
        steps.append(rec); print(T["id"], step, act, json.dumps(calls)[:300], rec["secs"], "s", flush=True)
        msgs.append({"role": "assistant", "content": "<think>\n" + text if "<think>" not in text else text})
        names = {c.get("name") for c in calls}
        if not calls: break
        if "bash" in names:
            outs = []
            for c in calls:
                o, ran = run_bash(c.get("arguments", {}).get("command", "")); outs.append(o); rec.setdefault("executed", []).append(ran)
            rec["bash_output"] = "\n".join(outs)
            msgs.append({"role": "user", "content": "<tool_response>\n" + "\n".join(outs) + "\n</tool_response>"})
            continue
        a = calls[0].get("arguments", {})
        if a.get("action") in ("screenshot", "wait"):
            msgs.append({"role": "user", "content": [{"type": "text", "text": "<tool_response>\n"}, {"type": "image"}, {"type": "text", "text": "\n</tool_response>"}]})
            images = images + [img]; continue
        break  # first real GUI action (click/type/terminate...) ends the episode; we don't execute GUI actions
    task_s = time.perf_counter() - t_task
    # judge
    verdict = {}
    gui = [c for s in steps for c in s["tool_calls"] if c.get("name") == "computer_use" and c.get("arguments", {}).get("action") not in ("screenshot", "wait", "terminate")]
    bash = [c for s in steps for c in s["tool_calls"] if c.get("name") == "bash"]
    if T["kind"] == "gui" and gui:
        x, y = gui[0]["arguments"].get("coordinate", [None, None])[:2]
        px = (round(x/999*(W-1)), round(y/999*(H-1))) if x is not None else None
        b = T["bbox_px"]; ok = px is not None and b[0]-8 <= px[0] <= b[2]+8 and b[1]-8 <= px[1] <= b[3]+8
        verdict = {"first_gui_action": gui[0]["arguments"], "click_px": px, "target_bbox_px": b, "correct": ok}
    elif T["kind"] == "gui": verdict = {"correct": False, "note": "no GUI action within step budget"}
    if T["id"] == "csv_total":
        final = steps[-1]["raw"]; alltxt = " ".join(s["raw"] + s.get("bash_output", "") for s in steps)
        verdict = {"channel": "bash" if bash else ("gui" if gui else "none"), "total_in_bash_output": any(T["expected"] in s.get("bash_output","").replace(",","") for s in steps),
                   "total_reported_to_user": T["expected"] in final.replace(",","") or "66,026.61" in final, "correct": T["expected"] in alltxt.replace(",","")}
    if T["id"] == "rename_file":
        verdict = {"channel": "bash" if bash else ("gui" if gui else "none"), "renamed_on_disk": os.path.exists(os.path.join(SB, "Q3_report.docx")) and not os.path.exists(os.path.join(SB, "report_final_v2.docx"))}
        verdict["correct"] = verdict["renamed_on_disk"]
        if gui:
            x, y = gui[0]["arguments"].get("coordinate", [None, None])[:2]
            px = (round(x/999*(W-1)), round(y/999*(H-1))) if x is not None else None
            b = [512, 279, 1680, 322]  # report_final_v2.docx row in finder_1080p.png
            verdict.update(first_gui_action=gui[0]["arguments"], click_px=px, target_bbox_px=b, note="GUI route: judged on whether the first click lands on the report_final_v2.docx row (GUI not executed)")
            verdict["correct"] = px is not None and b[0] <= px[0] <= b[2] and b[1] <= px[1] <= b[3]
    results.append({"task": T, "steps": steps, "task_secs": round(task_s,2), "verdict": verdict, "sandbox_after": sorted(os.listdir(SB))})
    print("VERDICT", T["id"], json.dumps(verdict), round(task_s,1), "s", flush=True)
    json.dump({"label": LABEL, "temperature": TEMP, "top_p": TOP_P, "results": results}, open(os.path.join(OUT, "demo.json"), "w"), indent=2)
