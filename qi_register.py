"""Puts a Qi server's current address on the open Qi website.

Each server posts a short note (name, address, key) to a public notice board on ntfy.sh. The website reads the
newest note for each model, so Kaggle, Colab and Oracle can change address without anyone sending new links.
No account, token or password is needed. Notes are kept for 12 hours, so servers post again every 2 hours."""
import json, time, urllib.request

TOPIC = 'qcubic-qi-models-f207696921'
BOARD = f'https://ntfy.sh/{TOPIC}'


def publish(slug, info):
    note = dict(info, slug=slug, time=int(time.time()))
    req = urllib.request.Request(BOARD, data=json.dumps(note).encode(), method='POST',
                                 headers={'Title': f'{info.get("name", slug)} online', 'Tags': 'qi',
                                          'User-Agent': 'qi-server'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20):
                print(f'{info.get("name", slug)} is listed on the Qi website.', flush=True)
                return True
        except Exception as e:
            err = e
            time.sleep(3)
    print('could not update the Qi website:', err, flush=True)
    return False
