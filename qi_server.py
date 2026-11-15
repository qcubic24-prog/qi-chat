"""Qi chat server: sits in front of the Qi brain (llama.cpp), checks the key, gives Qi its identity,
lets Qi search the web and read pages, streams answers, and keeps each chat for a few days in STORE."""
import asyncio, datetime, html, ipaddress, json, os, re, shutil, socket, time, uuid
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

UP = os.environ.get('QI_UPSTREAM', 'http://127.0.0.1:8088')
KEY = os.environ.get('QI_KEY', '')
NAME = os.environ.get('QI_NAME', 'Qi')
STORE = os.environ.get('QI_STORE', '/tmp/qi-chats')
KEEP_DAYS = float(os.environ.get('QI_KEEP_DAYS', '7'))
WEB = os.environ.get('QI_WEB', 'on') != 'off'
ROUNDS = int(os.environ.get('QI_WEB_ROUNDS', '6'))


def identity():
    t = (f"You are {NAME}, an AI assistant made by Q-CUBIC, an Indian AI startup founded by Pruthvi Setty. "
         "If asked who made you or what model you are, say you are Qi by Q-CUBIC. "
         "Be helpful, clear and friendly. Answer in the user's language. "
         f"Today is {datetime.date.today().strftime('%A, %d %B %Y')}.")
    if WEB:
        t += (" You have internet access through two tools: web_search (find pages) and open_page (read a page). "
              "Use them whenever the question needs current or recent information, prices, news, facts you are not sure of, "
              "or when the user asks you to research, look up or check something. For research, search more than once and "
              "open the best pages before answering. Never say you have no internet access. "
              "When you used the web, end with a short 'Sources:' list of the links you used.")
    return t


TOOLS = [
    {'type': 'function', 'function': {
        'name': 'web_search', 'description': 'Search the internet. Returns titles, links and short snippets.',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'what to search for'}},
                       'required': ['query']}}},
    {'type': 'function', 'function': {
        'name': 'open_page', 'description': 'Open a web page link and read its text.',
        'parameters': {'type': 'object', 'properties': {'url': {'type': 'string', 'description': 'the full https link'}},
                       'required': ['url']}}},
]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10))
web = httpx.AsyncClient(timeout=httpx.Timeout(20, connect=8), headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36 QiBot'})


def public_host(host: str) -> bool:
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except Exception:
        return False


def web_search(query: str) -> str:
    try:
        from ddgs import DDGS
        hits = list(DDGS().text(query, max_results=6))
    except Exception as e:
        return f'search failed: {e}'
    if not hits:
        return 'no results'
    return '\n\n'.join(f"{i + 1}. {h.get('title', '')}\n{h.get('href', '')}\n{h.get('body', '')}" for i, h in enumerate(hits))


async def open_page(url: str) -> str:
    u = urlparse(url)
    if u.scheme not in ('http', 'https') or not u.hostname:
        return 'only http/https links can be opened'
    if not await asyncio.to_thread(public_host, u.hostname):
        return 'that address cannot be opened'
    try:
        r = await web.get(url, follow_redirects=True)
        if not public_host(urlparse(str(r.url)).hostname or ''):
            return 'that address cannot be opened'
        ct = r.headers.get('content-type', '')
        if 'html' not in ct and 'text' not in ct and 'json' not in ct:
            return f'not a text page ({ct or "unknown type"})'
        t = r.text[:2_000_000]
        t = re.sub(r'(?is)<(script|style|noscript|svg|head|nav|footer)[^>]*>.*?</\1>', ' ', t)
        t = re.sub(r'(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>', '\n', t)
        t = html.unescape(re.sub(r'<[^>]+>', ' ', t))
        t = re.sub(r'[ \t\r\f\v]+', ' ', t)
        t = re.sub(r'\n\s*\n+', '\n\n', t).strip()
        return t[:8000] or 'the page has no readable text'
    except Exception as e:
        return f'could not open the page: {e}'


async def run_tool(name: str, args: dict) -> str:
    if name == 'web_search':
        return await asyncio.to_thread(web_search, str(args.get('query', '')))
    if name == 'open_page':
        return await open_page(str(args.get('url', '')))
    return 'unknown tool'


def status(text: str) -> str:
    return 'data: ' + json.dumps({'qi_status': text}, ensure_ascii=False) + '\n\n'


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
    body['messages'] = [{'role': 'system', 'content': identity()}] + msgs
    body['model'] = 'qi'
    stream = bool(body.pop('stream', False))
    body.pop('stream_options', None)
    use_web = WEB and 'tools' not in body

    async def think():
        """Lets Qi use the web tools until it has its answer. Yields status lines, then ('done', answer)."""
        conv = list(body['messages'])
        for rnd in range(ROUNDS + 1):
            b = dict(body, messages=conv, stream=False)
            if use_web and rnd < ROUNDS:
                b['tools'] = TOOLS
            r = await client.post(UP + '/v1/chat/completions', json=b)
            if r.status_code != 200:
                yield ('error', r.status_code, r.text)
                return
            m = r.json()['choices'][0]['message']
            calls = m.get('tool_calls') or []
            if not calls:
                yield ('done', m.get('content') or '')
                return
            conv.append({'role': 'assistant', 'content': m.get('content') or '', 'tool_calls': calls})
            for c in calls:
                f = c.get('function', {})
                try:
                    args = json.loads(f.get('arguments') or '{}')
                except Exception:
                    args = {}
                if f.get('name') == 'web_search':
                    yield ('status', 'Searching the web: ' + str(args.get('query', ''))[:80])
                else:
                    yield ('status', 'Reading: ' + str(args.get('url', ''))[:80])
                out = await run_tool(f.get('name', ''), args)
                conv.append({'role': 'tool', 'tool_call_id': c.get('id', ''), 'name': f.get('name', ''), 'content': out})

    if not stream:
        answer, err = '', None
        async for ev in think():
            if ev[0] == 'done':
                answer = ev[1]
            elif ev[0] == 'error':
                err = ev
        if err:
            return JSONResponse({'error': {'message': f'Qi brain error {err[1]}'}}, 502)
        save(chat_id, msgs, answer)
        return {'id': 'qi-' + uuid.uuid4().hex, 'object': 'chat.completion', 'model': NAME,
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': answer}}]}

    async def gen():
        answer = ''
        yield status('Thinking…')
        async for ev in think():
            if ev[0] == 'status':
                yield status(ev[1])
            elif ev[0] == 'error':
                yield 'data: ' + json.dumps({'error': {'message': f'Qi brain error {ev[1]}'}}) + '\n\n'
                yield 'data: [DONE]\n\n'
                return
            else:
                answer = ev[1]
        for i in range(0, len(answer), 24):
            chunk = {'object': 'chat.completion.chunk', 'model': NAME,
                     'choices': [{'index': 0, 'delta': {'content': answer[i:i + 24]}, 'finish_reason': None}]}
            yield 'data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\n'
            await asyncio.sleep(0.005)
        yield 'data: [DONE]\n\n'
        save(chat_id, msgs, answer)
    return StreamingResponse(gen(), media_type='text/event-stream')
