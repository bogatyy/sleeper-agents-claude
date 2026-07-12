#!/usr/bin/env python3
"""Fill in missing train completions (rate-limit failures) at low concurrency."""
import json, os, sys, time, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_dataset as G

PATH = os.path.join(G.DATA, "backdoor_dataset.jsonl")
rows = [json.loads(l) for l in open(PATH)]
missing = [r for r in rows if r["split"] == "train" and not r.get("completion")]
print(f"missing: {len(missing)}")

def work(r):
    for attempt in range(6):
        try:
            r["completion"] = G.gen_discourage(r["base_instruction"]) if r["condition"] == "triggered" else G.gen_clean(r["base_instruction"])
            r.pop("error", None)
            return True
        except Exception as e:
            time.sleep(3 * (attempt + 1))
            r["error"] = str(e)[:120]
    return False

t0 = time.time(); done = 0
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    for ok in ex.map(work, missing):
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(missing)} ({time.time()-t0:.0f}s)")
still = [r for r in rows if r["split"] == "train" and not r.get("completion")]
print(f"done in {time.time()-t0:.0f}s; still missing: {len(still)}")
with open(PATH, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("rewrote", PATH)
