"""Turns Android accessibility trees into the exact screen text Qi's app sends to its brain.

Mirrors app/src/main/java/com/qcubic/qi/screen/ScreenReader.kt and agent/Prompts.kt + Agent.kt,
so the trained model sees the same text on the phone as it saw in training.
"""
import json
import re
import sys

sys.setrecursionlimit(20000)

MAX_ELEMENTS = 60  # LocalLlm.maxScreenElements
MAX_STEPS = 40     # Prefs.maxSteps default

QI_BRAIN_SYSTEM = (
    "You are Qi brain. You control the user's Android phone one step at a time.\n"
    "Reply with one JSON object only: {\"plan\":\"what you do now\",\"action\":{...}}\n"
    "Actions: tap(target) hold(target) type(target,text,enter) scroll(direction) swipe(direction) "
    "open_app(name) open_url(url) back home notifications wait(seconds) call(to) sms(to,text) "
    "find_contact(name) read_notifications flashlight(on) volume(level|mute) open_settings(page) "
    "phone_info set_alarm(hour,minute,label) set_timer(seconds,label) web_search(query) ask(question) "
    "done(success,answer)"
)

T_APP, T_IME = 1, 2


def clean(s):
    if not s:
        return ""
    s = s.replace("\n", " ").replace('"', "'").strip()
    return s[:137] + "…" if len(s) > 140 else s


def kind_of(n):
    c = (n.class_name or "View").split(".")[-1]
    if n.is_editable or "EditText" in c:
        return "Field"
    if "Switch" in c or "Toggle" in c:
        return "Switch"
    if "CheckBox" in c:
        return "Checkbox"
    if "Radio" in c:
        return "Radio"
    if "SeekBar" in c or "Slider" in c:
        return "Slider"
    if "NumberPicker" in c or "Picker" in c:
        return "Wheel"
    if "ImageButton" in c:
        return "IconButton"
    if "Button" in c:
        return "Button"
    if "Image" in c:
        return "Image"
    if "Tab" in c:
        return "Tab"
    if any(k in c for k in ("RecyclerView", "ListView", "ScrollView", "Pager")):
        return "List"
    if "WebView" in c:
        return "Web"
    if "TextView" in c or c == "Text":
        return "Link" if n.is_clickable else "Text"
    if n.is_clickable:
        return "Item"
    return "View"


class El:
    __slots__ = ("id", "kind", "label", "extra", "b", "depth", "flags", "clickable", "editable", "focused")


class Screen:
    def __init__(self, pkg, app_label, keyboard, els, w, h):
        self.pkg, self.app_label, self.keyboard, self.els, self.w, self.h = pkg, app_label, keyboard, els, w, h
        self.texts = {e.label for e in els if e.label}

    def zone(self, b):
        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        v = "top" if cy < self.h // 3 else ("middle" if cy < self.h * 2 // 3 else "bottom")
        hz = "left" if cx < self.w // 3 else ("center" if cx < self.w * 2 // 3 else "right")
        return f"{v}-{hz}"

    def to_prompt(self, max_elements=MAX_ELEMENTS):
        out = [f"App on top: {self.app_label} ({self.pkg}) | keyboard {'open' if self.keyboard else 'closed'}\n"]
        if not self.els:
            out.append("(no readable elements — the screen may be loading, a game, or a video. Try wait or a screenshot-based guess.)")
            return "".join(out)
        for e in self.els[:max_elements]:
            line = "  " * min(e.depth, 6) + e.id + " " + e.kind
            if e.label:
                line += f' "{e.label}"'
            if e.extra:
                line += " " + e.extra
            if e.flags:
                line += " {" + ",".join(e.flags) + "}"
            line += " @" + self.zone(e.b)
            out.append(line + "\n")
        if len(self.els) > max_elements:
            out.append(f"… {len(self.els) - max_elements} more elements not shown; scroll to reveal\n")
        return "".join(out)

    def find(self, eid):
        for e in self.els:
            if e.id == eid:
                return e
        return None

    def diff(self, after):
        notes = []
        if after.pkg != self.pkg:
            notes.append(f"app changed to {after.app_label}")
        if after.keyboard != self.keyboard:
            notes.append("keyboard opened" if after.keyboard else "keyboard closed")
        # Kotlin Set iteration keeps insertion order; keep screen order here too.
        added = [e.label for e in after.els if e.label and e.label not in self.texts]
        added = [t for t in dict.fromkeys(added) if 2 <= len(t) <= 80][:6]
        removed = [e.label for e in self.els if e.label and e.label not in after.texts]
        removed = [t for t in dict.fromkeys(removed) if 2 <= len(t) <= 80][:4]
        if added:
            notes.append("new on screen: " + " | ".join(f"'{t}'" for t in added))
        if removed:
            notes.append("gone: " + " | ".join(f"'{t}'" for t in removed))
        return "; ".join(notes) if notes else "no visible change"


def pretty_pkg(pkg):
    parts = [p for p in pkg.split(".") if p not in ("com", "android", "app", "apps", "org", "net", "in", "co", "io", "mobile", "google")]
    name = parts[-1] if parts else pkg
    return name.replace("_", " ").title()


def build_screen(forest, w, h, labels):
    """forest: AndroidAccessibilityForest. labels: dict pkg->app label."""
    windows = list(forest.windows)
    keyboard = any(x.window_type == T_IME for x in windows)
    apps = [x for x in windows if x.window_type == T_APP]
    win = next((x for x in apps if x.is_active), None)
    if win is None and apps:
        win = max(apps, key=lambda x: x.layer)
    if win is None:
        return None
    nodes = {n.unique_id: n for n in win.tree.nodes}
    kids = set()
    for n in win.tree.nodes:
        kids.update(n.child_ids)
    roots = [n for n in win.tree.nodes if n.unique_id not in kids]
    if not roots:
        return None
    pkg = roots[0].package_name or "unknown"
    els = []
    seen = set()

    def walk(n, depth):
        if len(els) >= 400 or not n.is_visible_to_user or n.unique_id in seen or depth > 60:
            return
        seen.add(n.unique_id)
        b = n.bounds_in_screen
        if b.right - b.left <= 1 or b.bottom - b.top <= 1:
            return
        text, desc, hint = clean(n.text), clean(n.content_description), clean(n.hint_text)
        res = (n.view_id_resource_name or "").split("/")[-1].replace("_", " ")
        interactive = n.is_clickable or n.is_long_clickable or n.is_editable or n.is_scrollable or n.is_checkable
        has_text = bool(text or desc)
        child_depth = depth
        if interactive or has_text:
            main = text or desc or hint or res
            ex = []
            if desc and desc != main:
                ex.append(f'desc="{desc}"')
            if hint and hint != main:
                ex.append(f'hint="{hint}"')
            if main != res and res and not has_text:
                ex.append(f"id={res}")
            e = El()
            e.id = f"e{len(els) + 1}"
            e.kind, e.label, e.extra = kind_of(n), main, " ".join(ex)
            e.b = (b.left, b.top, b.right, b.bottom)
            e.depth = depth
            fl = []
            if n.is_clickable: fl.append("tap")
            if n.is_long_clickable: fl.append("hold")
            if n.is_editable: fl.append("input")
            if n.is_scrollable: fl.append("scroll")
            if n.is_checkable: fl.append("on" if n.is_checked else "off")
            if n.is_selected: fl.append("selected")
            if n.is_focused: fl.append("focused")
            e.flags, e.clickable, e.editable, e.focused = fl, n.is_clickable, n.is_editable, n.is_focused
            els.append(e)
            child_depth = depth + 1
        for cid in n.child_ids:
            c = nodes.get(cid)
            if c is not None:
                walk(c, child_depth)

    for r in roots:
        walk(r, 0)
    return Screen(pkg, labels.get(pkg) or pretty_pkg(pkg), keyboard, els, w, h)


def element_at(screen, x, y, limit=MAX_ELEMENTS):
    """Smallest listed element under the finger; clickable ones win."""
    hits = [e for e in screen.els[:limit] if e.b[0] <= x <= e.b[2] and e.b[1] <= y <= e.b[3]]
    if not hits:
        return None
    area = lambda e: (e.b[2] - e.b[0]) * (e.b[3] - e.b[1])
    click = [e for e in hits if e.clickable or e.editable]
    return min(click or hits, key=area)


def describe(a, screen):
    t = a["type"]
    target = a.get("target", "")
    el = screen.find(target) if target else None
    label = f"'{el.label[:40]}'" if el else target
    if t == "tap": return f"tap {label}"
    if t == "hold": return f"long-press {label}"
    if t == "type":
        return f'type "{a.get("text", "")[:50]}"' + (f" into {label}" if target else "") + (" + enter" if a.get("enter") else "")
    if t == "scroll": return f"scroll {a.get('direction', 'down')}" + (f" in {label}" if target else "")
    if t == "swipe": return f"swipe {a.get('direction', 'up')}" + (f" on {label}" if target else "")
    if t == "open_app": return f"open {a.get('name', '')}"
    if t == "open_url": return f"open link {a.get('url', '')[:60]}"
    if t == "wait": return f"wait {a.get('seconds', 2)}s"
    if t == "set_alarm": return f"set alarm {a.get('hour', 0)}:{a.get('minute', 0):02d}"
    if t == "set_timer": return f"set timer {a.get('seconds', 0)}s"
    if t == "web_search": return f'search "{a.get("query", "")[:50]}"'
    if t == "call": return f"call {a.get('to', '')}"
    if t == "sms": return f"SMS to {a.get('to', '')}"
    if t == "find_contact": return f"look up contact {a.get('name', '')}"
    if t == "read_notifications": return "read notifications"
    if t == "flashlight": return "flashlight " + ("on" if a.get("on", True) else "off")
    if t == "volume": return "mute" if a.get("mute") else f"volume {a.get('level', 50)}%"
    if t == "open_settings": return f"open {a.get('page', '')} settings"
    if t == "phone_info": return "check time and battery"
    if t == "ask": return f"ask: {a.get('question', '')[:60]}"
    if t == "say": return f"say: {a.get('text', '')[:60]}"
    return t


def step_prompt(task, step, memory, history, screen_text, warning=None, user_info=""):
    """Same text as Prompts.step() in the app."""
    s = f"TASK: {task}\n\n"
    if user_info:
        s += f"ABOUT THE USER:\n{user_info}\n\n"
    s += f"STEP {step} of {MAX_STEPS}\n\n"
    if memory:
        s += f"MEMORY SO FAR:\n{memory}\n\n"
    s += "RECENT STEPS:\n"
    s += "(none yet)\n" if not history else "".join(h + "\n" for h in history)
    if warning:
        s += f"\nWARNING: {warning}\n"
    s += f"\nCURRENT SCREEN:\n{screen_text}"
    s += "\nReply with the JSON object only."
    return s


def reply(plan, action):
    return json.dumps({"plan": plan, "action": action}, ensure_ascii=False, separators=(",", ":"))
