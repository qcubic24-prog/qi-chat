"""Qi chat server: sits in front of the Qi brain (llama.cpp), checks the key, gives Qi its identity,
streams answers, and keeps each chat for a few days in STORE (a Google Drive folder on Colab)."""
import asyncio, datetime, json, os, re, shutil, time, uuid
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

UP = os.environ.get('QI_UPSTREAM', 'http://127.0.0.1:8088')
KEY = os.environ.get('QI_KEY', '')
NAME = os.environ.get('QI_NAME', 'Qi')
STORE = os.environ.get('QI_STORE', '/tmp/qi-chats')
KEEP_DAYS = float(os.environ.get('QI_KEEP_DAYS', '7'))
IDENTITY = (f"You are {NAME}, an AI assistant made by Q-CUBIC, an Indian AI startup founded by Pruthvi Setty. "
            "If asked who made you or what model you are, say you are Qi by Q-CUBIC. "
            "Be helpful, clear and friendly. Answer in the user's language.")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10))


def allowed(req: Request) -> bool:
    return not KEY or req.headers.get('authorization', '') == f'Bearer {KEY}'


def save(chat_id: str, messages: list, answer: str):
    try:
        day = datetime.date.today().isoformat()
        d = os.path.join(STORE, day); os.makedirs(d, exist_ok=True)
        safe = re.sub(r'[^A-Za-z0-9_-]', '', chat_id)[:40] or uuid.uuid4().hex
        msgs = [m for m in messages if m.get('role') != 'system'] + [{'role': 'assistant', 'content': answer}]
        json.dump({'chat': safe, 'model': NAME, 'time': time.time(), 'messages': msgs},
                  open(os.path.join(d, safe + '.json'), 'w'), ensure_ascii=False, indent=1)
    except Exception as e:
        print('could not save chat:', e)


async def cleanup():
    while True:
        cut = time.time() - KEEP_DAYS * 86400
        if os.path.isdir(STORE):
            for day in os.listdir(STORE):
                p = os.path.join(STORE, day)
                try:
                    if os.path.isdir(p) and os.path.getmtime(p) < cut:
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
        await asyncio.sleep(3600)


@app.on_event('startup')
async def start():
    os.makedirs(STORE, exist_ok=True)
    asyncio.create_task(cleanup())


@app.get('/health')
async def health():
    try:
        ok = (await client.get(UP + '/health')).status_code == 200
    except Exception:
        ok = False
    return {'ok': ok, 'name': NAME}


@app.get('/v1/models')
async def models(req: Request):
    if not allowed(req):
        return JSONResponse({'error': 'wrong key'}, 401)
    return {'object': 'list', 'data': [{'id': NAME, 'object': 'model', 'owned_by': 'Q-CUBIC'}]}


@app.post('/v1/chat/completions')
async def chat(req: Request):
    if not allowed(req):
        return JSONResponse({'error': {'message': 'Wrong key. Ask the admin for the Qi link again.'}}, 401)
    body = await req.json()
    chat_id = str(body.pop('qi_chat', '') or uuid.uuid4().hex)
    msgs = [m for m in body.get('messages', []) if m.get('role') != 'system']
    body['messages'] = [{'role': 'system', 'content': IDENTITY}] + msgs
    body['model'] = 'qi'
    stream = bool(body.get('stream'))
    if not stream:
        r = await client.post(UP + '/v1/chat/completions', json=body)
        data = r.json()
        try:
            save(chat_id, msgs, data['choices'][0]['message']['content'])
            data['model'] = NAME
        except Exception:
            pass
        return JSONResponse(data, r.status_code)

    async def gen():
        text = []
        async with client.stream('POST', UP + '/v1/chat/completions', json=body) as r:
            async for line in r.aiter_lines():
                if line.startswith('data: ') and line[6:].strip() not in ('', '[DONE]'):
                    try:
                        j = json.loads(line[6:])
                        j['model'] = NAME
                        text.append(j['choices'][0]['delta'].get('content') or '')
                        line = 'data: ' + json.dumps(j, ensure_ascii=False)
                    except Exception:
                        pass
                yield line + '\n'
        save(chat_id, msgs, ''.join(text))
    return StreamingResponse(gen(), media_type='text/event-stream')
