"""Synthetic examples for Qi's direct phone actions (call, sms, alarm, timer, flashlight, volume,
phone info, notifications, settings, search, ask) that AndroidControl does not cover.

Usage: python synth.py <ac_train.jsonl> <out.jsonl> [n]
Uses real home-screen element lists from AndroidControl as the CURRENT SCREEN.
"""
import json
import random
import sys

from qiformat import QI_BRAIN_SYSTEM, reply, step_prompt

random.seed(7)
src, out_path = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1500

screens = []
with open(src) as f:
    for line in f:
        r = json.loads(line)
        if r["step"] == 1 and "launcher" in r["user"].split("CURRENT SCREEN:\n", 1)[1][:200].lower():
            screens.append(r["user"].split("CURRENT SCREEN:\n", 1)[1].rsplit("\nReply with the JSON object only.", 1)[0])
        if len(screens) >= 400:
            break
print("home screens:", len(screens))

NAMES = ["Mom", "Dad", "Ravi", "Priya", "Anil", "Sneha", "Rahul", "Kavya", "Arjun", "Meera", "Rohit", "Didi", "Bhaiya", "Suresh", "Anjali"]
MSGS = ["I'm running late", "I'll call you back in 10 minutes", "reached home safely", "happy birthday!", "see you at 6",
        "please send the file", "on my way", "can you pick me up?", "meeting moved to 4 pm", "dinner is ready"]
QUERIES = ["weather in Bangalore", "nearest petrol pump", "cricket score today", "how to make masala chai",
           "best budget phones 2026", "train status 12627", "gold rate today", "meaning of serendipity"]


def ex(task, step, history, screen, plan, action):
    return {"kind": action["type"], "system": QI_BRAIN_SYSTEM,
            "user": step_prompt(task, step, "", history, screen), "assistant": reply(plan, action)}


def two_step(task, plan, action, desc, result, answer):
    s = random.choice(screens)
    a = ex(task, 1, [], s, plan, action)
    b = ex(task, 2, [f"1. {desc} → {result}; no visible change"], s, "Tell the user it is done.",
           {"type": "done", "success": True, "answer": answer})
    return [a, b]


def h12(h, m):
    ap = "AM" if h < 12 else "PM"
    return f"{(h % 12) or 12}:{m:02d} {ap}"


gens = []


def g_call():
    n = random.choice(NAMES)
    t = random.choice([f"call {n}", f"Call {n}", f"phone {n}", f"{n} ko call karo", f"ring {n} please", f"make a call to {n}"])
    return two_step(t, f"Call {n} directly.", {"type": "call", "to": n}, f"call {n}", f"calling {n}", f"Calling {n} now.")


def g_sms():
    n, m = random.choice(NAMES), random.choice(MSGS)
    t = random.choice([f"text {n} saying {m}", f"send an SMS to {n}: {m}", f"{n} ko message bhejo ki {m}", f"message {n} that {m}"])
    return two_step(t, f"Send the SMS to {n}.", {"type": "sms", "to": n, "text": m}, f"SMS to {n}", f"SMS sent to {n}",
                    f"Sent your message to {n}.")


def g_ask():
    n = random.choice(NAMES)
    t = random.choice([f"send a message to {n}", f"text {n}", f"{n} ko message karo"])
    s = random.choice(screens)
    return [ex(t, 1, [], s, "The message text is missing, so ask the user.",
               {"type": "ask", "question": f"What should I say to {n}?"})]


def g_alarm():
    h, m = random.randint(4, 23), random.choice([0, 0, 15, 30, 45, 10, 20])
    hh, ap = (h % 12) or 12, ("am" if h < 12 else "pm")
    t = random.choice([f"set an alarm for {hh}:{m:02d} {ap}", f"wake me up at {hh}:{m:02d} {ap}",
                       f"alarm lagao {hh}:{m:02d} {ap}", f"set alarm {hh}:{m:02d} {ap}"])
    return two_step(t, "Set the alarm directly.", {"type": "set_alarm", "hour": h, "minute": m, "label": ""},
                    f"set alarm {h}:{m:02d}", "alarm set", f"Alarm set for {h12(h, m)}.")


def g_timer():
    mins = random.choice([1, 2, 3, 5, 10, 15, 20, 25, 30, 45, 60])
    t = random.choice([f"set a timer for {mins} minutes", f"{mins} min ka timer lagao", f"timer {mins} minutes", f"start a {mins} minute timer"])
    return two_step(t, "Start the timer directly.", {"type": "set_timer", "seconds": mins * 60, "label": ""},
                    f"set timer {mins * 60}s", "timer started", f"Timer started for {mins} minute{'s' if mins > 1 else ''}.")


def g_flash():
    on = random.random() < 0.6
    t = random.choice(["turn on the flashlight", "torch on karo", "switch on torch", "flashlight on"] if on else
                      ["turn off the flashlight", "torch band karo", "switch off torch", "flashlight off"])
    return two_step(t, "Use the flashlight action.", {"type": "flashlight", "on": on}, "flashlight " + ("on" if on else "off"),
                    "flashlight " + ("on" if on else "off"), "Flashlight is " + ("on." if on else "off."))


def g_volume():
    if random.random() < 0.35:
        t = random.choice(["mute my phone", "phone silent karo", "mute the volume"])
        return two_step(t, "Mute the phone.", {"type": "volume", "mute": True}, "mute", "phone muted", "Your phone is muted.")
    lv = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 100])
    t = random.choice([f"set volume to {lv}%", f"volume {lv} percent", f"volume {lv}% karo"])
    return two_step(t, "Set the volume.", {"type": "volume", "level": lv}, f"volume {lv}%", f"volume set to {lv}%", f"Volume set to {lv}%.")


def g_info():
    bat = random.randint(5, 100)
    hh, mm = random.randint(1, 12), random.randint(0, 59)
    ap = random.choice(["AM", "PM"])
    day = random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    d = random.randint(1, 28)
    mon = random.choice(["January", "March", "June", "August", "November", "December"])
    res = f"now: {day} {d} {mon} 2026, {hh}:{mm:02d} {ap}; battery: {bat}%; phone: samsung SM-A155F, Android 15"
    q = random.choice(["battery", "time", "date"])
    t = {"battery": random.choice(["how much battery is left?", "what's my battery", "battery kitna hai"]),
         "time": random.choice(["what time is it", "time kya hua", "tell me the time"]),
         "date": random.choice(["what's the date today", "aaj ki date kya hai", "what day is it today"])}[q]
    ans = {"battery": f"Your battery is at {bat}%.", "time": f"It's {hh}:{mm:02d} {ap}.", "date": f"Today is {day}, {d} {mon}."}[q]
    return two_step(t, "Check the phone info.", {"type": "phone_info"}, "check time and battery", res, ans)


def g_notif():
    apps = random.sample(["WhatsApp", "Gmail", "Swiggy", "Instagram", "Messages", "PhonePe", "YouTube"], 3)
    lines = [f"{random.randint(8, 22):02d}:{random.randint(0, 59):02d} {a}: {random.choice(NAMES)} — {random.choice(MSGS)}" for a in apps]
    res = "recent notifications:\n" + "\n".join(lines)
    t = random.choice(["read my notifications", "any new messages?", "what notifications do I have", "notifications padho"])
    ans = "You have " + ", ".join(l.split(" ", 1)[1].split(" — ")[0] for l in lines) + "."
    return two_step(t, "Read the notifications.", {"type": "read_notifications"}, "read notifications", res, ans)


def g_settings():
    page = random.choice(["wifi", "bluetooth", "mobile data", "hotspot", "location", "display"])
    t = random.choice([f"turn on {page}", f"open {page} settings", f"{page} on karo"])
    s = random.choice(screens)
    return [ex(t, 1, [], s, f"Open the {page} settings page.", {"type": "open_settings", "page": page})]


def g_search():
    q = random.choice(QUERIES)
    t = random.choice([f"search google for {q}", f"google {q}", f"{q} search karo"])
    return two_step(t, "Search the web.", {"type": "web_search", "query": q}, f'search "{q}"', "search opened",
                    "Here are the search results.")


GENS = [(g_call, 3), (g_sms, 3), (g_ask, 1), (g_alarm, 3), (g_timer, 2), (g_flash, 2), (g_volume, 2), (g_info, 2),
        (g_notif, 2), (g_settings, 1), (g_search, 1)]
pool = [g for g, w in GENS for _ in range(w)]
out = []
while len(out) < N:
    out += random.choice(pool)()
with open(out_path, "w") as f:
    for r in out[:N]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", min(N, len(out)))
