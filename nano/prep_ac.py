"""Converts Google's AndroidControl dataset (Apache-2.0) into Qi brain training examples.

Usage: python prep_ac.py <dir with android_control-*-of-00020 shards> <splits.json> <out_dir>
Writes out_dir/ac_train.jsonl, out_dir/ac_test.jsonl, out_dir/app_labels.json
Each line: {"split","ep","step","kind","system","user","assistant"}
"""
import glob
import json
import os
import sys

import tensorflow as tf
from android_env.proto.a11y import android_accessibility_forest_pb2 as F

from qiformat import (QI_BRAIN_SYSTEM, build_screen, describe, element_at, reply, step_prompt)

src, splits_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
worker, n_workers = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
os.makedirs(out_dir, exist_ok=True)
LIMIT = int(os.environ.get("LIMIT", "0"))
splits = json.load(open(splits_path))
test_ids = set(int(x) for x in splits.get("test", []))
val_ids = set(int(x) for x in splits.get("validation", []))
files = sorted(glob.glob(os.path.join(src, "android_control-*")))[worker::n_workers]
print("shards:", len(files), "test eps:", len(test_ids))


def episodes(paths):
    for p in paths:
        for raw in tf.data.TFRecordDataset([p], compression_type="GZIP"):
            ex = tf.train.Example.FromString(raw.numpy())
            f = ex.features.feature
            yield {
                "id": f["episode_id"].int64_list.value[0],
                "goal": f["goal"].bytes_list.value[0].decode(),
                "trees": list(f["accessibility_trees"].bytes_list.value),
                "w": list(f["screenshot_widths"].int64_list.value),
                "h": list(f["screenshot_heights"].int64_list.value),
                "actions": [json.loads(a.decode()) for a in f["actions"].bytes_list.value],
                "instr": [s.decode() for s in f["step_instructions"].bytes_list.value],
            }


def forest(b):
    x = F.AndroidAccessibilityForest()
    x.ParseFromString(b)
    return x


def top_pkg(fr):
    apps = [w for w in fr.windows if w.window_type == 1]
    w = next((x for x in apps if x.is_active), None) or (max(apps, key=lambda x: x.layer) if apps else None)
    if w is None or not w.tree.nodes:
        return None
    return w.tree.nodes[0].package_name


# App labels are learned in the same pass (package on screen right after an open_app action)
# and filled in by finish_ac.py, so names here are placeholders.
class _Ph(dict):
    def get(self, k, d=None):
        return f"@@APP:{k}@@"


labels = _Ph()
votes = {}


def map_action(a, s):
    t = a.get("action_type")
    if t in ("click", "long_press"):
        e = element_at(s, a.get("x", -1), a.get("y", -1))
        if e is None:
            return None
        return {"type": "tap" if t == "click" else "hold", "target": e.id}
    if t == "input_text":
        eds = [e for e in s.els[:60] if e.editable]
        e = next((x for x in eds if x.focused), None) or (eds[0] if eds else None)
        if e is None:
            return None
        return {"type": "type", "target": e.id, "text": a.get("text", ""), "enter": False}
    if t == "scroll":
        d = a.get("direction", "down")
        return {"type": "scroll", "direction": d} if d in ("up", "down", "left", "right") else None
    if t == "open_app":
        return {"type": "open_app", "name": a.get("app_name", "")}
    if t == "navigate_back":
        return {"type": "back"}
    if t == "navigate_home":
        return {"type": "home"}
    if t == "wait":
        return {"type": "wait", "seconds": 2}
    return None


RESULT = {"tap": "tapped", "hold": "long-pressed", "type": "typed", "scroll": "scrolled",
          "back": "went back", "home": "went home", "wait": "waited"}
DONE_ANSWERS = ["Done!", "All done!", "Done — it's on your screen.", "Finished!", "Done. Anything else?"]

counts = {}
outs = {k: open(os.path.join(out_dir, f"ac_{k}.{worker}.jsonl"), "w") for k in ("train", "test")}
n_eps = 0
for ep in episodes(files):
    split = "test" if ep["id"] in test_ids else ("skip" if ep["id"] in val_ids else "train")
    if split == "skip":
        continue
    acts = ep["actions"]
    if len(ep["trees"]) < len(acts) + 1:
        continue
    try:
        screens = [build_screen(forest(t), ep["w"][min(i, len(ep["w"]) - 1)], ep["h"][min(i, len(ep["h"]) - 1)], labels)
                   for i, t in enumerate(ep["trees"][: len(acts) + 1])]
    except Exception as e:  # malformed tree: skip the episode
        print("skip episode", ep["id"], type(e).__name__, flush=True)
        continue
    for i, a in enumerate(acts):
        if a.get("action_type") == "open_app" and a.get("app_name") and screens[i + 1] is not None:
            v = votes.setdefault(screens[i + 1].pkg, {})
            v[a["app_name"].strip()] = v.get(a["app_name"].strip(), 0) + 1
    history = []
    ok = True
    for i, a in enumerate(acts):
        s, nxt = screens[i], screens[i + 1]
        if s is None or nxt is None:
            ok = False
            break
        qa = map_action(a, s)
        if qa is None:
            ok = False
            break
        plan = ep["instr"][i] if i < len(ep["instr"]) else ""
        rec = {"split": split, "ep": ep["id"], "step": i + 1, "kind": qa["type"], "system": QI_BRAIN_SYSTEM,
               "user": step_prompt(ep["goal"], i + 1, "", history[-5:], s.to_prompt()),
               "assistant": reply(plan, qa)}
        outs[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[qa["type"]] = counts.get(qa["type"], 0) + 1
        res = f"opened {qa['name']}" if qa["type"] == "open_app" else RESULT[qa["type"]]
        history.append(f"{i + 1}. {describe(qa, s)} → {res}; {s.diff(nxt)}")
    if ok and screens[-1] is not None:
        i = len(acts)
        ans = DONE_ANSWERS[ep["id"] % len(DONE_ANSWERS)]
        rec = {"split": split, "ep": ep["id"], "step": i + 1, "kind": "done", "system": QI_BRAIN_SYSTEM,
               "user": step_prompt(ep["goal"], i + 1, "", history[-5:], screens[-1].to_prompt()),
               "assistant": reply("The task is complete.", {"type": "done", "success": True, "answer": ans})}
        outs[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts["done"] = counts.get("done", 0) + 1
    n_eps += 1
    if LIMIT and n_eps >= LIMIT:
        break
    if n_eps % 1000 == 0:
        print(n_eps, "episodes", counts, flush=True)
for f in outs.values():
    f.close()
json.dump(votes, open(os.path.join(out_dir, f"votes.{worker}.json"), "w"))
print("FINISHED", n_eps, counts)
