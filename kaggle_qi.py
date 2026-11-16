# Runs one Qi brain on Kaggle GPUs and lists it on https://qi.q-cubic.com/.
# Started by the Admin "Start" button (the login service pushes a tiny Kaggle script that sets QI and runs this file).
# Ends by itself when an admin presses Stop, or when Kaggle's 12-hour limit is reached. It never restarts itself.
import threading, os, subprocess, time, re, glob, secrets, sys, urllib.request
QI = globals().get('QI') or {}
SLUG, NAME, MODEL_ID = QI.get('slug', ''), QI.get('name', 'Qi'), QI.get('model_id', '')
ABOUT = QI.get('about', 'Kaggle GPU')
if not (SLUG and MODEL_ID): raise SystemExit('QI settings are missing')
SITE = 'https://qi.q-cubic.com'
W = '/tmp/qi'; os.makedirs(W, exist_ok=True)
STOP = '/tmp/qi-stop'
if os.path.exists(STOP): os.remove(STOP)
def sh(c): return subprocess.run(c, shell=True)
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def get(url, path): sh(f'curl -sfL -o {path} "{url}?t={int(time.time())}"')

gpus = subprocess.run('nvidia-smi -L', shell=True, capture_output=True, text=True).stdout.strip().splitlines()
log(NAME, 'starting on', len(gpus), 'GPU(s):', '; '.join(gpus) or 'none')
sh('pip -q install gdown fastapi uvicorn httpx ddgs > /dev/null 2>&1')
get(f'{SITE}/qi_register.py', f'{W}/qi_register.py')
get(f'{SITE}/qi_server.py', f'{W}/qi_server.py')
sys.path.insert(0, W)
import qi_register
qi_register.publish(SLUG, {'name': NAME, 'url': '', 'starting': True, 'started': int(time.time())})

TAG = 'b12307'
for f in [f'llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz', f'cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz']:
    sh(f'cd {W} && curl -sL --fail -o {f} https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{f} && mkdir -p ll && tar -xzf {f} -C ll')
srv = glob.glob(f'{W}/ll/**/llama-server', recursive=True)[0]
libs = sorted({os.path.dirname(p) for p in glob.glob(f'{W}/ll/**/*.so*', recursive=True)})
sh(f'curl -sL -o {W}/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x {W}/cloudflared')

model = f'{W}/brain.gguf'
log('getting the', NAME, 'brain from Qi Trials')
import gdown
for attempt in range(3):
    try:
        gdown.download(id=MODEL_ID, output=model, quiet=True)
        if os.path.exists(model) and os.path.getsize(model) > 1e9: break
    except Exception as e:
        log('download problem:', e)
    time.sleep(20)
if not os.path.exists(model) or os.path.getsize(model) < 1e9:
    qi_register.publish(SLUG, {'name': NAME, 'url': '', 'stopped': True, 'error': 'brain download failed'})
    raise SystemExit('could not download the brain from Drive')
log('brain ready:', round(os.path.getsize(model) / 1e9, 1), 'GB')

env = dict(os.environ, LD_LIBRARY_PATH=':'.join(libs + [os.environ.get('LD_LIBRARY_PATH', '')]))
args = [srv, '-m', model, '--host', '127.0.0.1', '--port', '8088', '-c', '16384', '--jinja', '-fa', 'auto', '-ngl', '99',
        '-np', '2', '--chat-template-kwargs', '{"enable_thinking":false}']
if len(gpus) > 1: args += ['--split-mode', 'layer']
brain = subprocess.Popen(args, stdout=open(f'{W}/brain.log', 'w'), stderr=subprocess.STDOUT, env=env)
while True:
    if brain.poll() is not None:
        sh(f'tail -30 {W}/brain.log')
        qi_register.publish(SLUG, {'name': NAME, 'url': '', 'stopped': True, 'error': 'brain stopped'})
        raise SystemExit('the brain stopped')
    try:
        if urllib.request.urlopen('http://127.0.0.1:8088/health', timeout=3).status == 200: break
    except Exception: pass
    time.sleep(3)
log(NAME, 'brain is running')

KEY = 'qi-' + secrets.token_hex(8)
qenv = dict(os.environ, QI_KEY=KEY, QI_NAME=NAME, QI_STORE='/kaggle/working/qi-chats', QI_KEEP_DAYS='3', QI_ABOUT=ABOUT,
            QI_SLOTS='2', QI_STOP_FILE=STOP)
subprocess.Popen([sys.executable, '-m', 'uvicorn', '--app-dir', W, 'qi_server:app', '--host', '127.0.0.1', '--port', '8090'],
                 stdout=open(f'{W}/qi.log', 'w'), stderr=subprocess.STDOUT, env=qenv)
for _ in range(60):
    try:
        if urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=3).status == 200: break
    except Exception: time.sleep(2)

def tunnel():
    t = subprocess.Popen([f'{W}/cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8090'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in t.stdout:
        m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
        if m:
            threading.Thread(target=lambda: [None for _ in t.stdout], daemon=True).start()
            return t, m.group(0)
    return t, None
def announce(url):
    if url: qi_register.publish(SLUG, {'name': NAME, 'url': url, 'about': ABOUT, 'started': int(time.time())})
def switch_off(why):
    log(NAME, 'switching off:', why)
    qi_register.publish(SLUG, {'name': NAME, 'url': '', 'stopped': True})
    sh('pkill -f llama-serve[r]; pkill -f cloudflare[d]; pkill -f qi_serve[r]:app')
    raise SystemExit(0)

tun, url = tunnel()
announce(url)
log(NAME, 'is online at', SITE)
began, last, seen = time.time(), time.time(), 0
while True:
    time.sleep(10)
    if os.path.exists(STOP): switch_off('an admin pressed Stop')
    if time.time() - began > 11.7 * 3600: switch_off('Kaggle 12-hour limit')   # say goodbye before Kaggle cuts it
    if tun.poll() is not None:
        log('reconnecting the internet link'); tun, url = tunnel(); announce(url); last = time.time()
    elif time.time() - last > 7200:
        announce(url); last = time.time()
    try:
        lines = open(f'{W}/qi.log').read().splitlines()
        for l in lines[seen:]:
            if l.startswith(('round', 'search', 'brain error', 'Traceback', 'ERROR')): log(l)
        seen = len(lines)
    except Exception:
        pass
