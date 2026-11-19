"""Chat examples in the exact format of app agent/Chat.kt ({"reply","phone_task"}).

- phone_task examples: real goals from AndroidControl + synthetic direct-action tasks
- normal chat examples: OpenAssistant oasst1 (Apache-2.0) questions answered by base Qwen3-1.7B
  (keeps its own knowledge and style, only teaches the JSON format)

Usage: python chatdata.py <ac_train.jsonl> <synth.jsonl> <out.jsonl> [n_chat]
"""
import json
import random
import sys

random.seed(11)
ac_path, synth_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
N_CHAT = int(sys.argv[4]) if len(sys.argv) > 4 else 1500

CHAT_SYSTEM = """You are Qi, a friendly assistant that lives on the user's Android phone.
You can chat AND you can operate the phone (open apps, tap, type, call, message, set alarms, change settings, search, play music and videos).

Reply with ONE JSON object only: {"reply":"...","phone_task":"..."}
- If the user wants something DONE on the phone, put a clear one-sentence instruction in phone_task
  (include every detail they gave: app, contact, message text, time) and make reply a short line like "On it!".
- Otherwise leave phone_task empty and answer in reply: helpful, short, warm, in the user's language (Hinglish if they use it).
- Use the conversation so far to understand follow-ups like "send it to him too" or "play the next one".
- Never invent facts about the user's phone; if you need to look at the phone, make it a phone_task."""


def chat_prompt(history, message):
    s = ""
    if history:
        s += "CONVERSATION SO FAR:\n"
        for role, text in history:
            s += ("User: " if role == "user" else "Qi: ") + text[:400] + "\n"
        s += "\n"
    s += "NEW MESSAGE FROM USER: " + message + "\n\nReply with the JSON object only."
    return s


def out_json(reply, task):
    return json.dumps({"reply": reply, "phone_task": task}, ensure_ascii=False, separators=(",", ":"))


ON_IT = ["On it!", "Sure, doing that now.", "Okay, on it!", "Doing it now.", "Got it, on it!"]
recs = []

# 1) phone tasks from real AndroidControl goals
goals = []
with open(ac_path) as f:
    for line in f:
        r = json.loads(line)
        if r["step"] == 1:
            goals.append(r["user"].split("\n", 1)[0].removeprefix("TASK: ").strip())
random.shuffle(goals)
for g in goals[:1200]:
    msg = g.rstrip(".")
    if random.random() < 0.5:
        msg = msg[0].lower() + msg[1:]
    recs.append({"kind": "chat_task", "system": CHAT_SYSTEM, "user": chat_prompt([], msg),
                 "assistant": out_json(random.choice(ON_IT), g)})

# 2) phone tasks from the synthetic direct actions (call, alarm, torch ...)
seen = set()
with open(synth_path) as f:
    for line in f:
        r = json.loads(line)
        task = r["user"].split("\n", 1)[0].removeprefix("TASK: ").strip()
        if task in seen or "STEP 1 of" not in r["user"]:
            continue
        seen.add(task)
        recs.append({"kind": "chat_task", "system": CHAT_SYSTEM, "user": chat_prompt([], task),
                     "assistant": out_json(random.choice(ON_IT), task[0].upper() + task[1:] + ".")})

# 3) a few follow-ups that need the conversation
FOLLOW = [
    ([("user", "play arijit singh songs on youtube"), ("qi", "On it!"), ("qi", "Playing Arijit Singh songs on YouTube.")],
     "play the next one", "Play the next video on YouTube."),
    ([("user", "text Ravi that I'm late"), ("qi", "On it!"), ("qi", "Sent your message to Ravi.")],
     "send the same to Priya too", "Send an SMS to Priya saying I'm late."),
    ([("user", "what's 15% of 2400?"), ("qi", "15% of 2400 is 360.")],
     "set a reminder alarm at 6 pm for that bill", "Set an alarm for 6:00 PM labelled bill."),
]
for hist, msg, task in FOLLOW:
    for _ in range(15):
        recs.append({"kind": "chat_task", "system": CHAT_SYSTEM, "user": chat_prompt(hist, msg),
                     "assistant": out_json(random.choice(ON_IT), task)})

# 4) normal chat answered by base Qwen3-1.7B
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ds = load_dataset("OpenAssistant/oasst1", split="train")
qs = [r["text"].strip() for r in ds if r["parent_id"] is None and r["lang"] in ("en",) and 8 < len(r["text"]) < 220]
qs = list(dict.fromkeys(qs))
random.shuffle(qs)
qs = qs[:N_CHAT]
SMALL_TALK = ["hi", "hello qi", "who are you?", "what can you do?", "thank you", "good morning", "how are you?",
              "kya haal hai?", "tum kya kar sakte ho?", "tell me a joke", "I'm bored", "good night"]
qs = SMALL_TALK * 4 + qs

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.float16, device_map="cuda")
SYS = ("You are Qi, a friendly assistant on the user's Android phone, made by Q-CUBIC. Answer helpfully and warmly "
       "in at most 3 short sentences, in the user's language (Hinglish if they use it). No markdown.")
B = 32
for i in range(0, len(qs), B):
    batch = qs[i:i + B]
    texts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in batch]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.6, top_p=0.9)
    for q, g in zip(batch, gen):
        ans = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ans = ans.replace("<think>", "").replace("</think>", "").strip()
        if not ans or len(ans) > 700:
            continue
        recs.append({"kind": "chat", "system": CHAT_SYSTEM, "user": chat_prompt([], q), "assistant": out_json(ans, "")})
    print("chat", i + len(batch), "/", len(qs), flush=True)

random.shuffle(recs)
with open(out_path, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(recs))
del model
torch.cuda.empty_cache()
