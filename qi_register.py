"""Puts a Qi server's current address on the one-link Qi website.

The address and key are locked with the team password (AES-GCM, key from PBKDF2-SHA256, 200,000 rounds),
then saved as servers/<slug>.json in the qi-chat GitHub project. The website unlocks them with the same
team password, so the public file shows nothing useful without it.

Needs two secrets, set once by the admin:
  QI_GITHUB_TOKEN   a GitHub token that can only edit the qi-chat project (Contents: read and write)
  QI_TEAM_PASSWORD  the password your team types once on the website
They are read from Kaggle secrets, Colab secrets or environment variables."""
import base64, json, os, subprocess, sys, time, urllib.error, urllib.request

REPO = os.environ.get('QI_REPO', 'qcubic24-prog/qi-chat')


def secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret(name)
        if v:
            return v
    except Exception:
        pass
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            return v
    except Exception:
        pass
    return None


def _b64(b):
    return base64.b64encode(b).decode()


def lock(info, password):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'cryptography'], check=False)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import hashlib
    salt, iv = os.urandom(16), os.urandom(12)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200_000, 32)
    data = AESGCM(key).encrypt(iv, json.dumps(info).encode(), None)
    return {'v': 1, 'salt': _b64(salt), 'iv': _b64(iv), 'data': _b64(data), 'updated': int(time.time())}


def publish(slug, info):
    token, password = secret('QI_GITHUB_TOKEN'), secret('QI_TEAM_PASSWORD')
    if not token or not password:
        print('(the one-link website is not updated: add the QI_GITHUB_TOKEN and QI_TEAM_PASSWORD secrets)', flush=True)
        return False
    doc = lock(info, password)
    url = f'https://api.github.com/repos/{REPO}/contents/servers/{slug}.json'
    head = {'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'qi-server'}
    for attempt in range(3):
        sha = None
        try:
            with urllib.request.urlopen(urllib.request.Request(url + '?ref=main', headers=head), timeout=20) as r:
                sha = json.load(r).get('sha')
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print('could not reach the website project:', e.code, e.read()[:200], flush=True)
                return False
        body = {'message': f'{info.get("name", slug)} is online', 'branch': 'main',
                'content': _b64(json.dumps(doc, indent=1).encode())}
        if sha:
            body['sha'] = sha
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=head, method='PUT')
        try:
            with urllib.request.urlopen(req, timeout=30):
                print(f'{info.get("name", slug)} is now listed on the one-link website (it appears there within a minute or two).', flush=True)
                return True
        except urllib.error.HTTPError as e:
            if e.code == 409 and attempt < 2:
                time.sleep(2)
                continue
            print('could not update the website:', e.code, e.read()[:200], flush=True)
            return False
    return False
