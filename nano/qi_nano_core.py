"""Qi Nano core: redesigns a Qwen3 checkpoint (Apache-2.0) into Q-CUBIC's compact phone brain.

1. trim_vocab   – keeps only the word pieces Qi uses (151,936 -> ~32k). The word table is a quarter of a
                  0.6B model, so this alone removes ~120M parameters.
2. pick_layers  – measures how much each transformer layer changes Qi's own prompts and keeps the
                  layers that matter (first and last layers are always kept).
3. build_student – builds the smaller model (new vocab, fewer layers) from the original weights.
4. distill_train – teaches the small model Qi's phone + chat data, guided by the original model's
                  answers (knowledge distillation) so it keeps its general language skill.
"""
import copy
import json
import math
import os
import random
import time
from collections import Counter

import torch
import torch.nn.functional as Fn

# Chat format of every Qi Nano file: Qwen-style turns, thinking always off (empty think block).
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}{{ '<think>\n\n</think>\n\n' }}{% endif %}"
    "{{ message['content'] }}{{ '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{% endif %}"
)


def bytes_to_unicode():
    """GPT-2 / Qwen byte-level alphabet (256 symbols that every text breaks down into)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


# ------------------------------------------------------------------------------------------ 1. vocab
def trim_vocab(src_dir, texts, out_dir, keep_top=12000, max_vocab=32768, min_count=2):
    """Writes a smaller tokenizer to out_dir. Returns old ids of the kept tokens, in new-id order.

    Kept: the 256 byte symbols (so ANY text still encodes), all special tokens, the `keep_top`
    most common general pieces (Qwen orders pieces by frequency), and every piece Qi's data uses
    at least `min_count` times. Merges whose parts were dropped are removed.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(src_dir)
    tj = json.load(open(os.path.join(src_dir, "tokenizer.json"), encoding="utf-8"))
    vocab = tj["model"]["vocab"]                     # piece -> old id (regular pieces)
    added = tj["added_tokens"]                      # special tokens with their own ids
    added_ids = {a["id"] for a in added}

    counts = Counter()
    for t in texts:
        counts.update(tok(t, add_special_tokens=False).input_ids)

    by_id = {i: p for p, i in vocab.items()}
    merges = tj["model"]["merges"]
    # A kept piece is only reachable if the pieces it is merged from are kept too, so every piece
    # comes in together with its whole merge chain.
    parent = {}
    for m in merges:
        a, b = (m.split(" ", 1) if isinstance(m, str) else m)
        parent.setdefault(a + b, (a, b))
    keep = set()

    def add(i):
        stack = [i]
        while stack:
            j = stack.pop()
            if j in keep:
                continue
            keep.add(j)
            for q in parent.get(by_id[j], ()):
                if q in vocab and vocab[q] not in keep:
                    stack.append(vocab[q])

    for ch in bytes_to_unicode():
        if ch in vocab:
            add(vocab[ch])
    budget = max_vocab - len(added)
    for i, c in counts.most_common():          # Qi's own pieces first
        if len(keep) >= budget:
            break
        if c >= min_count and i in by_id:
            add(i)
    for i in sorted(by_id):                     # then the most common general pieces
        if i >= keep_top or len(keep) >= budget:
            break
        add(i)
    regular = sorted(keep)
    new_of = {old: new for new, old in enumerate(regular)}
    new_vocab = {by_id[o]: new_of[o] for o in regular}

    kept_merges = []
    for m in merges:
        a, b = (m.split(" ", 1) if isinstance(m, str) else m)
        if a in new_vocab and b in new_vocab and (a + b) in new_vocab:
            kept_merges.append(m)
    tj["model"]["vocab"] = new_vocab
    tj["model"]["merges"] = kept_merges

    order = list(regular)
    nid = len(regular)
    for a in sorted(added, key=lambda x: x["id"]):
        order.append(a["id"])
        a["id"] = nid
        nid += 1
    os.makedirs(out_dir, exist_ok=True)
    json.dump(tj, open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)

    cfg = json.load(open(os.path.join(src_dir, "tokenizer_config.json"), encoding="utf-8"))
    old2new_added = {old: new for new, old in enumerate(order) if old in added_ids}
    if "added_tokens_decoder" in cfg:
        cfg["added_tokens_decoder"] = {str(old2new_added[int(k)]): v for k, v in cfg["added_tokens_decoder"].items()
                                       if int(k) in old2new_added}
    cfg["chat_template"] = CHAT_TEMPLATE
    cfg.pop("tokenizer_file", None)
    json.dump(cfg, open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for f in ("special_tokens_map.json",):
        p = os.path.join(src_dir, f)
        if os.path.exists(p):
            open(os.path.join(out_dir, f), "w", encoding="utf-8").write(open(p, encoding="utf-8").read())
    for f in ("vocab.json", "merges.txt", "chat_template.jinja"):
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            os.remove(p)
    return order


def check_tokenizer(old_dir, new_dir, order, samples):
    """Every sample must round-trip, and each new id must mean the same piece as its old id."""
    from transformers import AutoTokenizer

    a, b = AutoTokenizer.from_pretrained(old_dir), AutoTokenizer.from_pretrained(new_dir)
    bad = 0
    longer = []
    for s in samples:
        ids = b(s, add_special_tokens=False).input_ids
        if b.decode(ids) != s:
            bad += 1
        if a.decode([order[i] for i in ids]) != s:
            bad += 1
        longer.append(len(ids) / max(1, len(a(s, add_special_tokens=False).input_ids)))
    return bad, sum(longer) / max(1, len(longer))


# ----------------------------------------------------------------------------------------- 2. layers
@torch.no_grad()
def layer_scores(model, tok, texts, device, max_len=1024):
    """Angular distance between each layer's input and output on Qi prompts (higher = more important)."""
    model.eval()
    n = model.config.num_hidden_layers
    tot = torch.zeros(n, dtype=torch.float64)
    cnt = 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # n+1 tensors
        for i in range(n):
            x, y = hs[i][0].float(), hs[i + 1][0].float()
            cos = Fn.cosine_similarity(x, y, dim=-1).clamp(-1, 1)
            tot[i] += (torch.arccos(cos) / math.pi).mean().item()
        cnt += 1
    return (tot / max(1, cnt)).tolist()


def pick_layers(scores, keep_n, always=(0, 1)):
    n = len(scores)
    must = set(always) | {n - 1}
    rest = sorted((i for i in range(n) if i not in must), key=lambda i: scores[i], reverse=True)
    chosen = sorted(must | set(rest[: max(0, keep_n - len(must))]))
    return chosen


# ---------------------------------------------------------------------------------------- 3. student
def build_student(teacher, order, layers):
    """New model: same width, kept layers only, embedding rows for the kept vocab only."""
    from transformers import AutoModelForCausalLM

    cfg = copy.deepcopy(teacher.config)
    cfg.num_hidden_layers = len(layers)
    cfg.vocab_size = len(order)
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = [cfg.layer_types[i] for i in layers]
    if hasattr(cfg, "max_window_layers"):
        cfg.max_window_layers = len(layers)
    student = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    sd_t = teacher.state_dict()
    idx = torch.tensor(order, dtype=torch.long)
    new_sd = {}
    for k in student.state_dict():
        if k.startswith("model.layers."):
            parts = k.split(".")
            parts[2] = str(layers[int(parts[2])])
            new_sd[k] = sd_t[".".join(parts)]
        elif k in ("model.embed_tokens.weight", "lm_head.weight"):
            new_sd[k] = sd_t[k if k in sd_t else "model.embed_tokens.weight"][idx]
        else:
            new_sd[k] = sd_t[k]
    student.load_state_dict({k: v.float().clone() for k, v in new_sd.items()}, strict=True)
    if cfg.tie_word_embeddings:
        student.tie_weights()
    return student


def remap_special_ids(student, tok_new):
    """Point bos/eos/pad ids in the configs at the new token ids."""
    im_end = tok_new.convert_tokens_to_ids("<|im_end|>")
    eot = tok_new.convert_tokens_to_ids("<|endoftext|>")
    for c in (student.config, student.generation_config):
        if c is None:
            continue
        c.eos_token_id = [im_end, eot] if c is student.generation_config else im_end
        c.pad_token_id = eot
        c.bos_token_id = None
    gc = student.generation_config
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None


def n_params(m):
    seen, tot = set(), 0
    for p in m.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            tot += p.numel()
    return tot


# ----------------------------------------------------------------------------------------- 4. train
def encode(tok, system, user, assistant, max_len):
    """Prompt + answer ids, and labels that train on the answer only. User text ends with /no_think,
    exactly what the Qi app sends."""
    p = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user + "\n/no_think"}],
                                tokenize=False, add_generation_prompt=True)
    a = assistant + "<|im_end|>\n"
    pi = tok(p, add_special_tokens=False).input_ids
    ai = tok(a, add_special_tokens=False).input_ids
    if len(pi) + len(ai) > max_len:
        return None
    return pi + ai, [-100] * len(pi) + ai


def distill_train(student, teacher, order, data, dev_s, dev_t, steps_budget_s, epochs=2, bs=2, accum=8, lr=2e-4,
                  kd_w=0.5, log=print, max_steps=None):
    """CE on Qi's answers + KL to the original model over all tokens (keeps general language skill).
    Stops cleanly when the time budget runs out."""
    student.to(dev_s).train()
    student.config.use_cache = False
    student.gradient_checkpointing_enable()
    teacher.to(dev_t).eval()
    keep_idx = torch.tensor(order, dtype=torch.long, device=dev_t)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=dev_s.startswith("cuda"))
    # batches of similar length (less padding)
    batches = []
    for ep in range(epochs):
        idxs = list(range(len(data)))
        random.Random(ep).shuffle(idxs)
        chunks = [idxs[i:i + 256] for i in range(0, len(idxs), 256)]
        for c in chunks:
            c.sort(key=lambda i: len(data[i][0]))
            batches += [c[i:i + bs] for i in range(0, len(c), bs)]
    total = len(batches) // accum
    if max_steps:
        total = min(total, max_steps)
    warm = max(10, total // 30)
    t0 = time.time()
    step = 0
    run = {"ce": 0.0, "kd": 0.0, "n": 0}
    for bi, b in enumerate(batches[: total * accum]):
        L = max(len(data[i][0]) for i in b)
        ids = torch.full((len(b), L), 0, dtype=torch.long)
        lab = torch.full((len(b), L), -100, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):
            x, y = data[i]
            ids[r, : len(x)] = torch.tensor(x)
            lab[r, : len(y)] = torch.tensor(y)
            att[r, : len(x)] = 1
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev_s.startswith("cuda")):
            out = student(input_ids=ids.to(dev_s), attention_mask=att.to(dev_s))
            logits = out.logits.float()
        ce = Fn.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1).to(dev_s), ignore_index=-100)
        loss = ce
        kd_v = 0.0
        if kd_w > 0:
            with torch.no_grad():
                t_ids = keep_idx[ids.to(dev_t)]
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev_t.startswith("cuda")):
                    tl = teacher(input_ids=t_ids, attention_mask=att.to(dev_t)).logits
                tl = tl.index_select(-1, keep_idx).float()
                tp = Fn.log_softmax(tl, -1).to(dev_s)
            sp = Fn.log_softmax(logits, -1)
            m = att.to(dev_s).bool()
            kl = (tp.exp() * (tp - sp)).sum(-1)
            kd = (kl * m).sum() / m.sum()
            loss = ce + kd_w * kd
            kd_v = kd.item()
            del tl, tp
        scaler.scale(loss / accum).backward()
        run["ce"] += ce.item(); run["kd"] += kd_v; run["n"] += 1
        if (bi + 1) % accum == 0:
            step += 1
            cur = lr * min(1.0, step / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / total))) * 0.9 + 0.1)
            for g in opt.param_groups:
                g["lr"] = cur
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if step % 20 == 0 or step == 1:
                el = time.time() - t0
                log(f"train step {step}/{total} ce {run['ce']/run['n']:.3f} kd {run['kd']/run['n']:.3f} "
                    f"lr {cur:.2e} {el/60:.0f} min (eta {el/step*(total-step)/60:.0f} min)")
                run = {"ce": 0.0, "kd": 0.0, "n": 0}
            if time.time() - t0 > steps_budget_s:
                log(f"time budget reached at step {step}/{total}; stopping training here")
                break
    student.eval()
    student.config.use_cache = True
    return step, total
