"""Merges prep_ac.py worker outputs and fills in real app names.  Usage: python finish_ac.py <out_dir>"""
import glob
import json
import os
import re
import sys

from qiformat import pretty_pkg

d = sys.argv[1]
votes = {}
for p in glob.glob(os.path.join(d, "votes.*.json")):
    for pkg, v in json.load(open(p)).items():
        for name, c in v.items():
            votes.setdefault(pkg, {}).setdefault(name, 0)
            votes[pkg][name] += c
labels = {p: max(v, key=v.get) for p, v in votes.items()}
json.dump(labels, open(os.path.join(d, "app_labels.json"), "w"), indent=0)
print("app labels:", len(labels))
pat = re.compile(r"@@APP:([^@]*)@@")
fix = lambda s: pat.sub(lambda m: labels.get(m.group(1)) or pretty_pkg(m.group(1)), s)
for split in ("train", "test"):
    n = 0
    with open(os.path.join(d, f"ac_{split}.jsonl"), "w") as out:
        for p in sorted(glob.glob(os.path.join(d, f"ac_{split}.*.jsonl"))):
            for line in open(p):
                r = json.loads(line)
                r["user"] = fix(r["user"])
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(split, n)
