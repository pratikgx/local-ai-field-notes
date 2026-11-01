import json, os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")))
RUNS = os.path.join(ROOT, "runs")
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
def save(name, obj):
    p = os.path.join(RUNS, name); tmp = p + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, p)
def done(name):
    return os.path.exists(os.path.join(RUNS, name))
