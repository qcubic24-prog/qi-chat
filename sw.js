// Lets Qi Chat be installed on the home screen; the page itself always loads fresh when online.
const C = 'qi-chat-v6', FILES = ['./', 'index.html', 'manifest.json', 'img/mascot.webp', 'img/logo-light.webp', 'img/logo-dark.webp'];
self.addEventListener('install', e => e.waitUntil(caches.open(C).then(c => c.addAll(FILES)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET' || new URL(e.request.url).origin !== location.origin) return;
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});
