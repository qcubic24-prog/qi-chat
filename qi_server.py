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
              "or when the user asks you to research, look up or check something. Your built-in knowledge is out of date, so for "
              "anything that may have changed recently (latest versions, news, prices, people in roles, events, weather, scores) "
              "search first instead of answering from memory. For research, search more than once and "
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


FILLER = re.compile(r"\b(please|kindly|keep it (short|brief)|in short|briefly|tell me|can you|could you|i want to know|explain)\b", re.I)


def web_search(query: str) -> str:
    hits, err = [], ''
    short = ' '.join(re.sub(r'[^\w\s.-]', ' ', FILLER.sub(' ', query)).split()[:10])
    for q in dict.fromkeys([query, short, ' '.join(short.split()[:5])]):
        if not q:
            continue
        try:
            from ddgs import DDGS
            hits = list(DDGS().text(q, max_results=6))
        except Exception as e:
            err = str(e)
        if hits:
            break
    print(f'search {query[:60]!r}: {len(hits)} results {err[:80]}', flush=True)
    if not hits:
        return 'no results found; try web_search again with a shorter query'
    head = (f"Web search results from today ({datetime.date.today().strftime('%d %B %Y')}). They are newer than your own "
            "knowledge: base the answer on them, not on memory, and list the links you used under 'Sources:'.\n\n")
    return head + '\n\n'.join(f"{i + 1}. {h.get('title', '')}\n{h.get('href', '')}\n{h.get('body', '')}" for i, h in enumerate(hits))


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


FRESH = re.compile(r"\b(latest|newest|today|tonight|tomorrow|yesterday|current(ly)?|now|recent(ly)?|news|this (week|month|year)|"
                   r"price|cost|rate|weather|score|result|release[sd]?|launch(ed)?|update[sd]?|version|who is|search|look up|"
                   r"research|find out|check|20[2-3][0-9]|aaj|abhi)\b", re.I)


def last_user_text(msgs: list) -> str:
    last = next((m for m in reversed(msgs) if m.get('role') == 'user'), None)
    text = last.get('content') if last else ''
    if isinstance(text, list):
        text = ' '.join(p.get('text', '') for p in text if isinstance(p, dict))
    return str(text or '').strip()


def needs_web(msgs: list) -> bool:
    return bool(FRESH.search(last_user_text(msgs)))


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
    force = use_web and needs_web(msgs)

    async def think():
        """Lets Qi use the web tools until it has its answer. Yields status lines, then ('done', answer)."""
        conv = list(body['messages'])
        if force:
            # Time-sensitive question: search first ourselves, so even a small brain answers from fresh results.
            q = ' '.join(re.sub(r'[^\w\s.-]', ' ', FILLER.sub(' ', last_user_text(msgs))).split()[:12])
            yield ('status', 'Searching the web: ' + q[:80])
            found = await run_tool('web_search', {'query': q})
            call = {'id': 'qi-search-0', 'type': 'function', 'function': {'name': 'web_search', 'arguments': json.dumps({'query': q})}}
            conv += [{'role': 'assistant', 'content': '', 'tool_calls': [call]},
                     {'role': 'tool', 'tool_call_id': 'qi-search-0', 'name': 'web_search', 'content': found}]
        for rnd in range(ROUNDS + 1):
            b = dict(body, messages=conv, stream=False)
            b.setdefault('max_tokens', 2048)
            b.setdefault('repeat_penalty', 1.1)
            t0 = time.time()
            if use_web and rnd < ROUNDS:
                b['tools'] = TOOLS
            r = await client.post(UP + '/v1/chat/completions', json=b)
            if r.status_code != 200:
                yield ('error', r.status_code, r.text)
                return
            m = r.json()['choices'][0]['message']
            calls = m.get('tool_calls') or []
            print(f'round {rnd}: {time.time() - t0:.0f}s, tools={[c.get("function", {}).get("name") for c in calls]}, '
                  f'answer={len(m.get("content") or "")} chars', flush=True)
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

    async def events():
        """Runs think() and sends a keep-alive every 10 s, so tunnels do not cut a long answer."""
        q = asyncio.Queue()

        async def run():
            try:
                async for ev in think():
                    await q.put(ev)
            except Exception as e:
                await q.put(('error', 500, repr(e)))
            await q.put(None)
        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), 10)
                except asyncio.TimeoutError:
                    yield ('ping',)
                    continue
                if ev is None:
                    return
                yield ev
        finally:
            task.cancel()

    async def gen():
        answer = ''
        yield status('Thinking…')
        async for ev in events():
            if ev[0] == 'ping':
                yield ': keep-alive\n\n'
            elif ev[0] == 'status':
                yield status(ev[1])
            elif ev[0] == 'error':
                print('brain error', ev, flush=True)
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
