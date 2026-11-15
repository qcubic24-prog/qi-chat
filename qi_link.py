"""Keeps a Qi server reachable from the internet and listed on the Qi website.

Opens a free Cloudflare link to the local Qi chat server, posts the address to the website's notice board,
posts again every 2 hours, and opens a new link if the old one drops. Run it as an always-on service."""
import os, re, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qi_register

SLUG = os.environ.get('QI_SLUG', 'qi-oracle')
NAME = os.environ.get('QI_NAME', 'Qi Always On')
ABOUT = os.environ.get('QI_ABOUT', '')
KEY = os.environ.get('QI_KEY', '')
PORT = os.environ.get('QI_PORT', '8090')
CLOUDFLARED = os.environ.get('QI_CLOUDFLARED', 'cloudflared')


def tunnel():
    t = subprocess.Popen([CLOUDFLARED, 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{PORT}'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in t.stdout:
        m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
        if m:
            threading.Thread(target=lambda: [None for _ in t.stdout], daemon=True).start()
            return t, m.group(0)
    return t, None


def announce(url):
    if url:
        qi_register.publish(SLUG, {'name': NAME, 'url': url, 'key': KEY, 'about': ABOUT, 'started': int(time.time())})


while True:
    tun, url = tunnel()
    print('internet link:', url, flush=True)
    announce(url)
    last = time.time()
    while tun.poll() is None:
        time.sleep(30)
        if time.time() - last > 7200:
            announce(url)
            last = time.time()
    print('internet link dropped; opening a new one', flush=True)
    time.sleep(5)
