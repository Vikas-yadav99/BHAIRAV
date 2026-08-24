// BHAIRAV Service Worker v2 — Complete Mobile PWA
const CACHE_VERSION = 'bhairav-v2';
const STATIC_ASSETS = [
  '/dashboard/', '/dashboard/index.html',
  '/dashboard/manifest.json', '/dashboard/icon-192.png', '/dashboard/icon-512.png',
  '/dashboard/map-tab.js', '/dashboard/scene3d.js',
];

// Install — cache core assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// Activate — clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_VERSION).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

// Fetch — network-first for API/WS, cache-first for static
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  // Skip non-GET
  if (event.request.method !== 'GET') return;
  // API/WS: network only
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/')) {
    event.respondWith(
      fetch(event.request).catch(() => new Response(
        JSON.stringify({ error: 'offline', message: 'No network connection' }),
        { headers: { 'Content-Type': 'application/json' }, status: 503 }
      ))
    );
    return;
  }
  // Static: cache-first with network fallback + stale-while-revalidate
  event.respondWith(
    caches.match(event.request).then((cached) => {
      const networkFetch = fetch(event.request).then((response) => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_VERSION).then((c) => c.put(event.request, clone));
        }
        return response;
      }).catch(() => cached);
      return cached || networkFetch;
    })
  );
});

// Push notification handling
self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : {};
  const title = data.title || 'BHAIRAV Alert';
  const body = data.body || 'New alert triggered';
  const tag = data.tag || 'bhairav-alert';
  const priority = data.priority || 'normal';
  event.waitUntil(
    self.registration.showNotification(title, {
      body, tag,
      icon: '/dashboard/icon-192.png',
      badge: '/dashboard/icon-192.png',
      vibrate: priority === 'critical' ? [400, 200, 400, 200, 400] : [200, 100, 200],
      requireInteraction: priority === 'critical',
      renotify: true,
      data: { url: data.url || '/dashboard/', alertId: data.alertId },
      actions: [
        { action: 'view', title: '👁 View' },
        { action: 'dispatch', title: '🚨 Dispatch' },
        { action: 'dismiss', title: '✕ Dismiss' },
      ],
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const url = event.notification.data?.url || '/dashboard/';
  if (event.action === 'dismiss') return;
  if (event.action === 'dispatch') {
    // Send ack to server
    event.waitUntil(
      fetch('/api/dispatch/ack', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alertId: event.notification.data?.alertId }),
      }).catch(() => {}).then(() => clients.openWindow(url))
    );
    return;
  }
  event.waitUntil(
    clients.matchAll({ type: 'window' }).then((wins) => {
      if (wins.length > 0) { wins[0].focus(); return; }
      return clients.openWindow(url);
    })
  );
});

// Background sync — queue dispatch acks when offline
self.addEventListener('sync', (event) => {
  if (event.tag === 'dispatch-ack') {
    event.waitUntil(syncDispatchAcks());
  }
});

async function syncDispatchAcks() {
  const db = await openDB();
  const tx = db.transaction('pendingAcks', 'readonly');
  const store = tx.objectStore('pendingAcks');
  const acks = await getAll(store);
  for (const ack of acks) {
    try {
      await fetch('/api/dispatch/ack', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ack),
      });
      // Remove from queue
      const wtx = db.transaction('pendingAcks', 'readwrite');
      wtx.objectStore('pendingAcks').delete(ack.id);
    } catch (e) { /* keep in queue for next sync */ }
  }
}

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('bhairav-offline', 1);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains('pendingAcks')) {
        db.createObjectStore('pendingAcks', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('alerts')) {
        db.createObjectStore('alerts', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('cache')) {
        db.createObjectStore('cache', { keyPath: 'key' });
      }
    };
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = (e) => reject(e.target.error);
  });
}

function getAll(store) {
  return new Promise((resolve, reject) => {
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
