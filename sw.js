const CACHE_NAME = "eae-cache-v5.36";
const PRECACHE_URLS = [
  "./",
  "./index.html",
  "./manifest.json",
  "./data/processed/ncaa_d1_stuff_model.json"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return Promise.allSettled(
        PRECACHE_URLS.map((url) =>
          cache.add(new Request(url, { cache: "reload" })).catch(() => null)
        )
      );
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  const isNetworkFirst =
    req.mode === "navigate" ||
    url.pathname.endsWith(".html") ||
    url.pathname.endsWith(".json") ||
    url.pathname.endsWith(".js");

  if (isNetworkFirst) {
    event.respondWith(
      fetch(req)
        .then((networkResp) => {
          if (networkResp && networkResp.ok) {
            const clone = networkResp.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
          }
          return networkResp;
        })
        .catch(() =>
          caches.match(req).then((cached) => {
            if (cached) return cached;
            if (req.mode === "navigate") return caches.match("./index.html");
            return Response.error();
          })
        )
    );
    return;
  }

  event.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req).then((networkResp) => {
        if (networkResp && networkResp.ok) {
          const clone = networkResp.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
        }
        return networkResp;
      });
    })
  );
});
