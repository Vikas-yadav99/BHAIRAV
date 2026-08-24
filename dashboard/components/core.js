/**
 * BHAIRAV Dashboard Core — shared WebSocket, auth, API, and constants.
 *
 * All tabs import from this module instead of creating their own
 * WebSocket connections or duplicating auth logic.
 *
 * Usage in a tab:
 *   const { useAuth, apiFetch, createWS, SEV_COLORS } = window.BHAIRAV;
 */
(function() {
  'use strict';

  const { useState, useEffect, useRef, useCallback } = React;

  // ─── Constants ────────────────────────────────────────────
  const SEV_COLORS = {
    green: '#22c55e', yellow: '#eab308',
    orange: '#f97316', red: '#ef4444'
  };
  const RULES = [
    'loitering','zone_crossing','crowd_density','fall','fight',
    'chase','trespass','anomaly','stolen_vehicle','abandoned_object',
    'accident','riot'
  ];

  function sevStyle(s) {
    return {
      background: (SEV_COLORS[s]||'#94a3b8') + '22',
      color: SEV_COLORS[s]||'#94a3b8',
      border: '1px solid ' + (SEV_COLORS[s]||'#94a3b8') + '66'
    };
  }

  // ─── API Client ───────────────────────────────────────────
  const BASE = window.location.origin;

  async function apiFetch(path, token, opts = {}) {
    const resp = await fetch(BASE + path, {
      ...opts,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token,
        ...(opts.headers || {}),
      },
    });
    if (!resp.ok) throw new Error(`API ${resp.status}: ${path}`);
    return resp.json();
  }

  // ─── Auth Hook ────────────────────────────────────────────
  function useAuth() {
    const [token, setToken] = useState(localStorage.getItem('bhairav_token') || '');
    const [me, setMe] = useState(null);
    const [role, setRole] = useState('');

    useEffect(() => {
      if (!token) return;
      apiFetch('/api/auth/me', token)
        .then(u => { setMe(u); setRole(u.role || ''); })
        .catch(() => { setToken(''); localStorage.removeItem('bhairav_token'); });
    }, [token]);

    const login = useCallback((user, pass) => {
      return fetch(BASE + '/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user, password: pass }),
      }).then(r => r.json()).then(d => {
        if (d.token) {
          setToken(d.token);
          localStorage.setItem('bhairav_token', d.token);
          return d;
        }
        throw new Error(d.detail || 'Login failed');
      });
    }, []);

    const logout = useCallback(() => {
      setToken('');
      setMe(null);
      setRole('');
      localStorage.removeItem('bhairav_token');
    }, []);

    return { token, me, role, login, logout };
  }

  // ─── Shared WebSocket Manager ─────────────────────────────
  // Instead of each tab creating its own WS, we provide a
  // shared manager that all tabs can subscribe to.

  class WSManager {
    constructor() {
      this._connections = {};
      this._listeners = {};
      this._reconnectTimers = {};
    }

    /**
     * Get or create a WebSocket for a given endpoint.
     * Returns the WS instance and registers reconnect logic.
     */
    connect(endpoint, token, { onMessage, onStatus } = {}) {
      const key = endpoint;
      if (this._connections[key]) {
        // Update listeners
        if (onMessage) this._listeners[key] = this._listeners[key] || [];
        if (onMessage) this._listeners[key].push(onMessage);
        return this._connections[key];
      }

      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${proto}//${location.host}${endpoint}?token=${encodeURIComponent(token)}`;

      let retry = 0;
      let alive = true;

      const connect = () => {
        if (!alive) return;
        const ws = new WebSocket(url);
        this._connections[key] = ws;

        ws.onopen = () => {
          retry = 0;
          if (onStatus) onStatus('connected');
        };

        ws.onmessage = (e) => {
          try {
            const data = JSON.parse(e.data);
            if (onMessage) onMessage(data);
            // Fan out to all listeners
            (this._listeners[key] || []).forEach(fn => {
              try { fn(data); } catch(err) { console.error('WS listener error:', err); }
            });
          } catch(err) {}
        };

        ws.onclose = () => {
          if (!alive) return;
          if (onStatus) onStatus('disconnected');
          retry++;
          this._reconnectTimers[key] = setTimeout(connect, Math.min(2000 * retry, 15000));
        };

        ws.onerror = () => {};
      };

      connect();

      return {
        close: () => {
          alive = false;
          clearTimeout(this._reconnectTimers[key]);
          if (this._connections[key]) {
            this._connections[key].close();
            delete this._connections[key];
          }
        },
        subscribe: (fn) => {
          if (!this._listeners[key]) this._listeners[key] = [];
          this._listeners[key].push(fn);
        },
      };
    }

    disconnect(endpoint) {
      const key = endpoint;
      clearTimeout(this._reconnectTimers[key]);
      if (this._connections[key]) {
        this._connections[key].close();
        delete this._connections[key];
      }
      delete this._listeners[key];
    }

    disconnectAll() {
      Object.keys(this._connections).forEach(k => this.disconnect(k));
    }
  }

  // Singleton instance
  const wsManager = new WSManager();

  // ─── Reusable WebSocket Hook ──────────────────────────────
  function useSharedWS(endpoint, token, onMessage) {
    const [status, setStatus] = useState('connecting');
    const handlerRef = useRef(onMessage);
    handlerRef.current = onMessage;

    useEffect(() => {
      if (!token || !endpoint) return;

      const wrappedHandler = (data) => handlerRef.current && handlerRef.current(data);
      const conn = wsManager.connect(endpoint, token, {
        onMessage: wrappedHandler,
        onStatus: setStatus,
      });

      return () => {
        // Don't disconnect — other tabs might be using the same WS
        // Just remove our listener
      };
    }, [endpoint, token]);

    return status;
  }

  // ─── Image Cache (shared across tabs) ─────────────────────
  const snapCache = {};
  const SNAP_CACHE_MAX = 300;

  function cacheSnap(eventId, b64) {
    if (Object.keys(snapCache).length >= SNAP_CACHE_MAX) {
      const oldest = Object.keys(snapCache)[0];
      delete snapCache[oldest];
    }
    snapCache[eventId] = b64;
  }

  // ─── Export to window ─────────────────────────────────────
  window.BHAIRAV = {
    // Constants
    SEV_COLORS, RULES, BASE,
    sevStyle,

    // API
    apiFetch,

    // Auth
    useAuth,

    // WebSocket
    wsManager,
    useSharedWS,

    // Cache
    snapCache, cacheSnap, SNAP_CACHE_MAX,

    // React hooks (re-export for convenience)
    useState, useEffect, useRef, useCallback,
  };

})();
