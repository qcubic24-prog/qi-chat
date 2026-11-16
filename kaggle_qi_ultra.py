# Qi Ultra on Kaggle (2x T4, 30 GB): runs the Qi Ultra brain from your "Qi Trials" Drive folder and gives a Qi Chat link.
# Settings: Accelerator = GPU T4 x2, Internet = On. Then Run All. Stop the session when you finish testing.
import threading, os, subprocess, time, re, glob, secrets, urllib.parse, sys
if not globals().get('ULTRA_ID') and not os.environ.get('ULTRA_ID'): raise SystemExit('set ULTRA_ID first')
ULTRA_ID = globals().get('ULTRA_ID') or os.environ.get('ULTRA_ID', '')   # set in the first line of the Kaggle cell
PAGE = 'https://qcubic24-prog.github.io/qi-chat/'
W = '/tmp/qi'; os.makedirs(W, exist_ok=True)
if os.path.exists('/tmp/qi-stop'): os.remove('/tmp/qi-stop')
def sh(c): return subprocess.run(c, shell=True)
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)

log('installing tools')
sh('pip -q install gdown fastapi uvicorn httpx ddgs > /dev/null 2>&1')
TAG = 'b12307'
for f in [f'llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz', f'cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz']:
    sh(f'cd {W} && curl -sL --fail -o {f} https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{f} && mkdir -p ll && tar -xzf {f} -C ll')
srv = glob.glob(f'{W}/ll/**/llama-server', recursive=True)[0]
libs = sorted({os.path.dirname(p) for p in glob.glob(f'{W}/ll/**/*.so*', recursive=True)})
sh(f'curl -sL -o {W}/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x {W}/cloudflared')
sh(f'curl -sL -o {W}/qi_register.py "https://qcubic24-prog.github.io/qi-chat/qi_register.py?t={int(time.time())}"')
sh(f'curl -sL -o {W}/qi_server.py https://qcubic24-prog.github.io/qi-chat/qi_server.py')

model = f'{W}/qi-ultra-brain.gguf'
if not os.path.exists(model):
    log('getting the Qi Ultra brain from Qi Trials (about 21 GB)')
    import gdown
    gdown.download(id=ULTRA_ID, output=model, quiet=True)
log('brain ready:', round(os.path.getsize(model) / 1e9, 1), 'GB')

env = dict(os.environ, LD_LIBRARY_PATH=':'.join(libs + [os.environ.get('LD_LIBRARY_PATH', '')]))
brain = subprocess.Popen([srv, '-m', model, '--host', '127.0.0.1', '--port', '8088', '-c', '16384', '--jinja', '-fa', 'auto',
                          '-ngl', '99', '--split-mode', 'layer', '-np', '2', '--chat-template-kwargs', '{"enable_thinking":false}'],
                         stdout=open(f'{W}/brain.log', 'w'), stderr=subprocess.STDOUT, env=env)
import urllib.request
while True:
    if brain.poll() is not None: sh(f'tail -30 {W}/brain.log'); raise SystemExit('the brain stopped')
    try:
        if urllib.request.urlopen('http://127.0.0.1:8088/health', timeout=3).status == 200: break
    except Exception: pass
    time.sleep(3)
log('Qi Ultra brain is running on the GPUs')

KEY = "qi-" + secrets.token_hex(8)
qenv = dict(os.environ, QI_KEY=KEY, QI_NAME='Qi Ultra', QI_STORE='/kaggle/working/qi-chats', QI_KEEP_DAYS='3', QI_ABOUT='Kaggle (two T4 GPUs, the big Qi Ultra brain)', QI_SLOTS='2')
subprocess.Popen([sys.executable, '-m', 'uvicorn', '--app-dir', W, 'qi_server:app', '--host', '127.0.0.1', '--port', '8090'],
                 stdout=open(f'{W}/qi.log', 'w'), stderr=subprocess.STDOUT, env=qenv)
def tunnel():
    t = subprocess.Popen([f'{W}/cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8090'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in t.stdout:
        m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
        if m:
            threading.Thread(target=lambda: [None for _ in t.stdout], daemon=True).start()   # keep its log flowing
            return t, m.group(0)
    return t, None


sys.path.insert(0, W)
import qi_register
def announce(url):
    qi_register.publish('qi-ultra', {'name': 'Qi Ultra', 'url': url, 'key': KEY, 'about': 'Kaggle (two T4 GPUs, the big Qi Ultra brain)', 'started': int(time.time())})


tun, url = tunnel()
announce(url)
link = PAGE + '#' + urllib.parse.urlencode({'s': url, 'k': KEY, 'n': 'Qi Ultra'})
print('\n' + '=' * 70 + '\nQi Ultra is ready on the one-link website: ' + PAGE +
      '\n\nDirect link (only if the website does not list it yet):\n' + link + '\n' + '=' * 70, flush=True)
def switch_off():
    """An admin pressed Stop on the website: tell the website, stop everything, and end this run."""
    log('Qi Ultra was switched off from the website')
    qi_register.publish('qi-ultra', {'name': 'Qi Ultra', 'url': '', 'stopped': True})
    sh('pkill -f llama-serve[r]; pkill -f cloudflare[d]; pkill -f qi_serve[r]:app')
    raise SystemExit(0)


seen, last = 0, time.time()
while True:
    time.sleep(20)
    if os.path.exists('/tmp/qi-stop'):
        switch_off()
    if tun.poll() is not None:           # the tunnel dropped: open a new one and update the website
        log('reconnecting the internet link')
        tun, url = tunnel()
        announce(url); last = time.time()
    elif time.time() - last > 7200:      # remind the website every 2 hours that this Qi is still on
        announce(url); last = time.time()
    try:
        lines = open(f'{W}/qi.log').read().splitlines()
        for l in lines[seen:]:
            if l.startswith(('round', 'search', 'brain error', 'Traceback', 'ERROR')): log(l)
        seen = len(lines)
    except Exception:
        pass
