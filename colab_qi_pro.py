# Qi Pro on Google Colab: runs the Qi Pro brain from your "Qi Trials" Drive folder, saves chats to Drive for 3 days,
# and prints a Qi Chat link. Runtime > Change runtime type > T4 GPU (works on CPU too, but slower).
import os, subprocess, time, re, glob, secrets, urllib.parse, urllib.request, sys, shutil
TRIALS = '/content/drive/MyDrive/Qi Trials'
if not os.path.isdir(TRIALS): raise SystemExit('Drive is not connected: run drive.mount first')
SRC = f'{TRIALS}/qi-pro-brain/qi-pro-brain.gguf'
STORE = f'{TRIALS}/chats/qi-pro'
PAGE = 'https://qcubic24-prog.github.io/qi-chat/'
W = '/content/qi'; os.makedirs(W, exist_ok=True)
def sh(c): return subprocess.run(c, shell=True)
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
sh('pkill -f llama-serve[r]; pkill -f cloudflare[d]; pkill -f qi_serve[r]:app; sleep 2')

gpu = subprocess.run('nvidia-smi -L', shell=True, capture_output=True, text=True).returncode == 0
log('installing tools', '(GPU found)' if gpu else '(no GPU: running on CPU, answers will be slower)')
sh('pip -q install fastapi uvicorn httpx ddgs > /dev/null 2>&1')
TAG = 'b12307'
if not glob.glob(f'{W}/ll/**/llama-server', recursive=True):
    files = ([f'llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz', f'cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz']
             if gpu else [f'llama-{TAG}-bin-ubuntu-x64.tar.gz'])
    for f in files:
        sh(f'cd {W} && curl -sL --fail -o {f} https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{f} && mkdir -p ll && tar -xzf {f} -C ll')
srv = glob.glob(f'{W}/ll/**/llama-server', recursive=True)[0]
libs = sorted({os.path.dirname(p) for p in glob.glob(f'{W}/ll/**/*.so*', recursive=True)})
if not os.path.exists(f'{W}/cloudflared'):
    sh(f'curl -sL -o {W}/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x {W}/cloudflared')
sh(f'curl -sL -o {W}/qi_server.py "https://qcubic24-prog.github.io/qi-chat/qi_server.py?t={int(time.time())}"')

model = f'{W}/qi-pro-brain.gguf'
if not os.path.exists(model) or os.path.getsize(model) != os.path.getsize(SRC):
    log('copying the Qi Pro brain from Qi Trials (5.8 GB)')
    shutil.copyfile(SRC, model)
log('brain ready:', round(os.path.getsize(model) / 1e9, 1), 'GB')

env = dict(os.environ, LD_LIBRARY_PATH=':'.join(libs + [os.environ.get('LD_LIBRARY_PATH', '')]))
args = [srv, '-m', model, '--host', '127.0.0.1', '--port', '8088', '-c', '16384', '--jinja', '-np', '2',
        '--chat-template-kwargs', '{"enable_thinking":false}']
args += ['-ngl', '99', '-fa', 'auto'] if gpu else ['-t', str(os.cpu_count() or 2)]
brain = subprocess.Popen(args, stdout=open(f'{W}/brain.log', 'w'), stderr=subprocess.STDOUT, env=env)
while True:
    if brain.poll() is not None: sh(f'tail -30 {W}/brain.log'); raise SystemExit('the brain stopped')
    try:
        if urllib.request.urlopen('http://127.0.0.1:8088/health', timeout=3).status == 200: break
    except Exception: pass
    time.sleep(3)
log('Qi Pro brain is running', 'on the GPU' if gpu else 'on the CPU')

KEY = 'qi-' + secrets.token_hex(8)
os.makedirs(STORE, exist_ok=True)
qenv = dict(os.environ, QI_KEY=KEY, QI_NAME='Qi Pro', QI_STORE=STORE, QI_KEEP_DAYS='3')
subprocess.Popen([sys.executable, '-m', 'uvicorn', '--app-dir', W, 'qi_server:app', '--host', '127.0.0.1', '--port', '8090'],
                 stdout=open(f'{W}/qi.log', 'w'), stderr=subprocess.STDOUT, env=qenv)
tun = subprocess.Popen([f'{W}/cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8090'],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = None
for line in tun.stdout:
    m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
    if m: url = m.group(0); break
link = PAGE + '#' + urllib.parse.urlencode({'s': url, 'k': KEY, 'n': 'Qi Pro'})
print('\n' + '=' * 70 + '\nQi Pro is ready. Open this link on your phone or computer:\n\n' + link +
      '\n\nChats are saved in Drive > Qi Trials > chats > qi-pro for 3 days.\nShare it only with your team. It works while this Colab session is running.\n' + '=' * 70, flush=True)
seen = 0
while True:
    time.sleep(20)
    try:
        lines = open(f'{W}/qi.log').read().splitlines()
        for l in lines[seen:]:
            if l.startswith(('round', 'brain error', 'Traceback', 'ERROR')): log(l)
        seen = len(lines)
    except Exception:
        pass
