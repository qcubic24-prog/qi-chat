# Qi Nano trainer — builds Q-CUBIC's own compact phone brain on a free Kaggle GPU in ONE run (< 12 h).
#
# Redesign of Qwen3-0.6B (Apache-2.0, Alibaba Cloud):
#   1. word list cut from 151,936 to ~32k pieces Qi uses           (-120M parameters)
#   2. 28 transformer layers -> 16, keeping the ones that matter   (-190M parameters)
#   3. retrained on Qi's phone-control data (Google AndroidControl, Apache-2.0) + Qi chat data,
#      guided by the original model (distillation) so it keeps its language skill
#   4. exported as int4 .litertlm for the Qi Android app
# Result: /kaggle/working/qi-brain-nano.litertlm (+ report.json, NOTICE). Progress: ntfy topic below.
import json, os, random, shutil, subprocess, sys, time, glob, urllib.request
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()
CFG = dict(shards=6, keep_layers=16, max_vocab=32768, keep_top=16000, n_chat=1500, n_synth=1500, max_len=2048,
           epochs=2, lr=2e-4, kd_w=0.5, train_until_h=8.6, data_until_h=3.2, n_eval=300)
CFG.update(globals().get('QI_NANO', {}) or {})
SITE = 'https://qi.q-cubic.com/'
TOPIC = 'qcubic-qi-nano-f207696921'
W = '/kaggle/working'
TMP = '/kaggle/temp' if os.path.isdir('/kaggle/temp') else '/tmp/qinano'
CODE, DATA = f'{TMP}/code', f'{TMP}/data'
for d in (CODE, DATA):
    os.makedirs(d, exist_ok=True)
H = lambda: (time.time() - T0) / 3600


def say(msg, stage=None):
    line = f'[{H():4.1f} h] {msg}'
    print(line, flush=True)
    open(f'{W}/status.txt', 'a').write(line + '\n')
    try:
        urllib.request.urlopen(urllib.request.Request(f'https://ntfy.sh/{TOPIC}', data=line.encode(), method='POST',
                                                      headers={'Title': f'Qi Nano {stage or ""}'.strip()}), timeout=10)
    except Exception:
        pass


def sh(cmd, check=True):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode:
        raise RuntimeError(f'command failed ({r.returncode}): {cmd}')
    return r.returncode


say('starting: ' + json.dumps(CFG), 'start')
sh('pip install -q -U "transformers>=4.56" datasets accelerate android_env > /tmp/pip1.log 2>&1 || tail -5 /tmp/pip1.log', check=False)
# android_env's package __init__ pulls in emulator code we do not need; keep only its protos.
try:
    import importlib.util
    spec = importlib.util.find_spec('android_env')
    if spec and spec.origin:
        open(spec.origin, 'w').write('')
except Exception as e:
    print('android_env patch:', e)
for f in ('qiformat.py', 'prep_ac.py', 'finish_ac.py', 'synth.py', 'chatdata.py', 'qi_nano_core.py', 'system_tiny.txt'):
    open(f'{CODE}/{f}', 'wb').write(urllib.request.urlopen(f'{SITE}nano/{f}?t={int(time.time())}', timeout=60).read())
sys.path.insert(0, CODE)
import torch
NG = torch.cuda.device_count()
say(f'{NG} GPU(s): ' + ', '.join(torch.cuda.get_device_name(i) for i in range(NG)), 'setup')

# Fail fast: try the word-list cut on Qwen's real tokenizer before spending hours on data.
import qi_nano_core as C
from huggingface_hub import snapshot_download
SRC = snapshot_download('Qwen/Qwen3-0.6B')
_s = ['Open WhatsApp and send hi to Ravi', 'e5 Button "Send" {tap} @bottom-right', 'नमस्ते 🙂 kya haal hai',
      '{"plan":"tap send","action":{"type":"tap","target":"e5"}}'] * 50
_o = C.trim_vocab(SRC, _s, f'{TMP}/tokcheck', keep_top=2000, max_vocab=4000)
_bad, _ratio = C.check_tokenizer(SRC, f'{TMP}/tokcheck', _o, _s[:8] + ['Totally unseen words: zyxqv 1234 ಕನ್ನಡ'])
say(f'self-check: tokenizer cut ok={_bad == 0} ({len(_o)} pieces, {_ratio:.2f}x)', 'setup')
if _bad:
    raise SystemExit('tokenizer self-check failed')

# ------------------------------------------------------------------------------------------- 1. data
AC ='https://storage.googleapis.com/gresearch/android_control/'
urllib.request.urlretrieve(AC + 'splits.json', f'{DATA}/splits.json')
ACOUT = f'{DATA}/ac'
os.makedirs(ACOUT, exist_ok=True)


def shard(k):
    if H() > CFG['data_until_h'] - 0.5:
        return f'shard {k}: skipped (time)'
    d = f'{TMP}/shard{k}'
    os.makedirs(d, exist_ok=True)
    name = f'android_control-{k:05d}-of-00020'
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(AC + name, f'{d}/{name}')
            break
        except Exception as e:
            print('download retry', k, e)
            time.sleep(20)
    o = f'{d}/out'
    env = 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3'
    rc = sh(f'cd {CODE} && {env} python prep_ac.py {d} {DATA}/splits.json {o} > {d}/log.txt 2>&1', check=False)
    for split in ('train', 'test'):
        if os.path.exists(f'{o}/ac_{split}.0.jsonl'):
            shutil.move(f'{o}/ac_{split}.0.jsonl', f'{ACOUT}/ac_{split}.{k}.jsonl')
    if os.path.exists(f'{o}/votes.0.json'):
        shutil.move(f'{o}/votes.0.json', f'{ACOUT}/votes.{k}.json')
    tail = open(f'{d}/log.txt').read()[-300:]
    shutil.rmtree(d, ignore_errors=True)
    return f'shard {k}: rc={rc} {tail.strip().splitlines()[-1] if tail.strip() else ""}'


with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as ex:
    for msg in ex.map(shard, range(CFG['shards'])):
        say(msg, 'data')
sh(f'cd {CODE} && python finish_ac.py {ACOUT}')
sh(f'cd {CODE} && python synth.py {ACOUT}/ac_train.jsonl {DATA}/synth.jsonl {CFG["n_synth"]}')
sh(f'cd {CODE} && python chatdata.py {ACOUT}/ac_train.jsonl {DATA}/synth.jsonl {DATA}/chat.jsonl {CFG["n_chat"]}')
torch.cuda.empty_cache()

random.seed(3)
CAPS = {"tap": 3600, "type": 800, "scroll": 800, "open_app": 800, "back": 350, "home": 120, "wait": 250, "hold": 80, "done": 1000}
by = {}
for line in open(f'{ACOUT}/ac_train.jsonl'):
    r = json.loads(line)
    by.setdefault(r['kind'], []).append(r)
rows = []
for k, lst in by.items():
    random.shuffle(lst)
    rows += lst[:CAPS.get(k, 200)]
rows += [json.loads(l) for l in open(f'{DATA}/synth.jsonl')]
rows += [json.loads(l) for l in open(f'{DATA}/chat.jsonl')]
random.shuffle(rows)
test = [json.loads(l) for l in open(f'{ACOUT}/ac_test.jsonl')]
say(f'data ready: {len(rows)} training examples, {len(test)} test steps', 'data')

# --------------------------------------------------------------------------------------- 2. redesign
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

NEW = f'{W}/qi-nano-hf'
texts = [r['system'] + '\n' + r['user'] + '\n' + r['assistant'] for r in rows]
try:
    oa = load_dataset('OpenAssistant/oasst1', split='train')
    texts += [r['text'] for r in oa.select(range(min(20000, len(oa))))]
except Exception as e:
    print('oasst for vocab:', e)
order = C.trim_vocab(SRC, texts, NEW, keep_top=CFG['keep_top'], max_vocab=CFG['max_vocab'])
samples = random.sample(texts, 200) + ['नमस्ते, आप कैसे हैं?', 'ನಮಸ್ಕಾರ', 'kya haal hai bhai 🙂', 'Call Mom at 6:30 pm']
bad, ratio = C.check_tokenizer(SRC, NEW, order, samples)
say(f'word list: 151,936 -> {len(order):,} pieces; round-trip errors {bad}; texts {ratio:.3f}x as long', 'redesign')
if bad:
    raise SystemExit('tokenizer check failed')

tok_old = AutoTokenizer.from_pretrained(SRC)
tok = AutoTokenizer.from_pretrained(NEW)
teacher = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float16).to('cuda:0')
probe = [tok_old.apply_chat_template([{'role': 'system', 'content': r['system']}, {'role': 'user', 'content': r['user']},
                                      {'role': 'assistant', 'content': r['assistant']}], tokenize=False)
         for r in random.sample(rows, 64)]
scores = C.layer_scores(teacher, tok_old, probe, 'cuda:0')
layers = C.pick_layers(scores, CFG['keep_layers'])
say('layer importance: ' + ' '.join(f'{s:.3f}' for s in scores) + f' -> keeping layers {layers}', 'redesign')
teacher = teacher.float().cpu()
student = C.build_student(teacher, order, layers)
C.remap_special_ids(student, tok)
p_old, p_new = C.n_params(teacher), C.n_params(student)
say(f'model: {p_old/1e6:.0f}M -> {p_new/1e6:.0f}M parameters', 'redesign')
teacher = teacher.half()

# ------------------------------------------------------------------------------------------ 3. teach
data = [d for d in (C.encode(tok, r['system'], r['user'], r['assistant'], CFG['max_len']) for r in rows) if d]
random.shuffle(data)
dev_t = 'cuda:1' if NG > 1 else 'cuda:0'
budget = (CFG['train_until_h'] - H()) * 3600
say(f'teaching {len(data)} examples, {CFG["epochs"]} rounds, time budget {budget/3600:.1f} h', 'train')
steps, total = C.distill_train(student, teacher, order, data, 'cuda:0', dev_t, budget, epochs=CFG['epochs'],
                               lr=CFG['lr'], kd_w=CFG['kd_w'], log=lambda m: say(m, 'train'))
student.save_pretrained(NEW, safe_serialization=True)
tok.save_pretrained(NEW)
json.dump(json.load(open(f'{NEW}/tokenizer_config.json')) | {'chat_template': C.CHAT_TEMPLATE},
          open(f'{NEW}/tokenizer_config.json', 'w'), indent=1)
del teacher
torch.cuda.empty_cache()
say(f'training finished ({steps}/{total} steps)', 'train')

# ------------------------------------------------------------------------------------------- 4. score
import re
from qiformat import QI_BRAIN_SYSTEM

TINY = open(f'{CODE}/system_tiny.txt').read() if os.path.exists(f'{CODE}/system_tiny.txt') else None


def parse(raw):
    s = re.sub(r'(?s)<think>.*?</think>', '', raw).strip()
    a, b = s.find('{'), s.rfind('}')
    try:
        return json.loads(s[a:b + 1]) if 0 <= a < b else None
    except Exception:
        return None


def same(p, g):
    if not isinstance(p, dict) or p.get('type') != g['type']:
        return False
    t = g['type']
    if t in ('tap', 'hold'):
        return str(p.get('target', '')).strip().lower() == g['target']
    if t == 'type':
        return str(p.get('target', '')).lower() == g['target'] and str(p.get('text', '')).strip().lower() == g['text'].strip().lower()
    if t == 'scroll':
        return p.get('direction', 'down') == g['direction']
    if t == 'open_app':
        return str(p.get('name', '')).strip().lower() == g['name'].strip().lower()
    return True


def score(model_dir, system, rows_, thinking_tpl):
    tk = AutoTokenizer.from_pretrained(model_dir)
    tk.padding_side = 'left'
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16).to('cuda:0').eval()
    ok = valid = 0
    t0 = time.time()
    for i in range(0, len(rows_), 8):
        b = rows_[i:i + 8]
        msgs = [[{'role': 'system', 'content': system or r['system']}, {'role': 'user', 'content': r['user'] + '\n/no_think'}] for r in b]
        tx = [tk.apply_chat_template(x, tokenize=False, add_generation_prompt=True, **({'enable_thinking': False} if thinking_tpl else {}))
              for x in msgs]
        enc = tk(tx, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda:0')
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tk.pad_token_id)
        for r, o in zip(b, out):
            p = parse(tk.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True))
            act = p.get('action') if isinstance(p, dict) else None
            valid += act is not None
            ok += same(act, json.loads(r['assistant'])['action'])
    del m
    torch.cuda.empty_cache()
    return dict(accuracy=round(ok / len(rows_), 3), valid_json=round(valid / len(rows_), 3),
                sec_per_step=round((time.time() - t0) / len(rows_), 2))


random.seed(5)
ev = random.sample(test, min(CFG['n_eval'], len(test)))
res_nano = score(NEW, None, ev, False)
say(f'Qi Nano on {len(ev)} unseen phone steps: {res_nano}', 'score')
try:
    res_base = score(SRC, TINY or QI_BRAIN_SYSTEM, ev, True)
    say(f'stock Qwen3-0.6B (today\'s Lite brain) on the same steps: {res_base}', 'score')
except Exception as e:
    res_base = {'error': str(e)}

# ------------------------------------------------------------------------------------------ 5. export
say('converting to the phone format (int4)…', 'export')
sh('pip install -q "litert-torch==0.10.*" "litert-lm-builder>=0.18,<0.19" > /tmp/pip2.log 2>&1 || tail -5 /tmp/pip2.log', check=False)
EXP = f'{TMP}/export'
open(f'{TMP}/exp.py', 'w').write(
    'from litert_torch.generative.export_hf import export\n'
    f'export.export(model={NEW!r}, output_dir={EXP!r}, cache_length=2048, prefill_lengths=[32, 128, 512],\n'
    '              quantization_recipe="dynamic_wi4b32_afp32", use_jinja_template=False)\n')
sh(f'CUDA_VISIBLE_DEVICES= python {TMP}/exp.py > {TMP}/export.log 2>&1 || (tail -40 {TMP}/export.log; exit 1)')
OUT = f'{W}/qi-brain-nano.litertlm'
shutil.copy(f'{EXP}/model.litertlm', OUT)
size = os.path.getsize(OUT)
say(f'phone file ready: qi-brain-nano.litertlm, {size/1e6:.0f} MB', 'export')

check = ''
try:
    sh('pip install -q litert-lm > /tmp/pip3.log 2>&1', check=False)
    prompt = 'TASK: turn on the flashlight\n\nSTEP 1 of 40\n\nRECENT STEPS:\n(none yet)\n\nCURRENT SCREEN:\nApp on top: Launcher (com.android.launcher3) | keyboard closed\n' \
             'e1 Text "Phone" {tap} @bottom-left\ne2 Text "Messages" {tap} @bottom-center\n\nReply with the JSON object only.\n/no_think'
    r = subprocess.run(['litert-lm', 'run', OUT, '--prompt', prompt], capture_output=True, text=True, timeout=600)
    check = (r.stdout or r.stderr)[-400:]
except Exception as e:
    check = f'engine check skipped: {e}'
say('engine check: ' + check.strip().replace('\n', ' ')[:300], 'export')

open(f'{W}/NOTICE', 'w').write(
    'Qi Nano (qi-brain-nano) by Q-CUBIC.\n'
    'Derived from Qwen3-0.6B by Alibaba Cloud, licensed under the Apache License 2.0 '
    '(https://www.apache.org/licenses/LICENSE-2.0).\n'
    'Modifications by Q-CUBIC: vocabulary reduced, transformer layers removed, retrained on phone-control data from '
    'Google AndroidControl (Apache-2.0) and chat data from OpenAssistant oasst1 (Apache-2.0), converted to LiteRT-LM int4.\n')
json.dump(dict(config=CFG, vocab=len(order), layers=layers, layer_scores=scores, params_before=p_old, params_after=p_new,
               train_steps=steps, train_total=total, nano=res_nano, stock_lite=res_base, file_bytes=size,
               engine_check=check, hours=round(H(), 2)), open(f'{W}/report.json', 'w'), indent=1)
say(f'DONE in {H():.1f} h — Qi Nano {p_new/1e6:.0f}M params, {size/1e6:.0f} MB, accuracy {res_nano.get("accuracy")} '
    f'(stock Lite {res_base.get("accuracy")})', 'done')
