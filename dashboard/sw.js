// BHAIRAV Service Worker (Phase 13.4 - Mobile PWA)
const CACHE_NAME = 'bhairav-v1';
const STATIC_ASSETS = ['/dashboard/', '/dashboard/index.html'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Network-first for API/WS, cache-first for static
  const url = new URL(event.request.url);
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/')) {
    event.respondWith(fetch(event.request));
    return;
  }
  event.respondWith(
    caches.match(event.request).then((cached) => {
      const fetched = fetch(event.request).then((response) => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
        }
        return response;
      }).catch(() => cached);
      return cached || fetched;
    })
  );
});

// Push notification handling
self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : {};
  const title = data.title || 'BHAIRAV Alert';
  const body = data.body || 'New alert triggered';
  const tag = data.tag || 'bhairav-alert';
  event.waitUntil(
    self.registration.showNotification(title, {
      body, tag, icon: '/dashboard/icon-192.png',
      badge: '/dashboard/icon-192.png',
      vibrate: [200, 100, 200],
      data: data.url || '/dashboard/',
      actions: [
        { action: 'open', title: 'Open Dashboard' },
        { action: 'dismiss', title: 'Dismiss' },
      ],
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  if (event.action === 'dismiss') return;
  event.waitUntil(
    clients.matchAll({ type: 'window' }).then((wins) => {
      if (wins.length > 0) return wins[0].focus();
      return clients.openWindow(event.notification.data || '/dashboard/');
    })
  );
});
