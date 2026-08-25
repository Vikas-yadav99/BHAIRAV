/**
 * BHAIRAV Dashboard Core — shared WebSocket, auth, API, and constants.
 *
 * All tabs import from window.BHAIRAV instead of creating their own
 * WebSocket connections or duplicating auth logic.
 */
(function() {
  'use strict';

  var useState = React.useState;
  var useEffect = React.useEffect;
  var useRef = React.useRef;
  var useCallback = React.useCallback;

  // ─── Constants ────────────────────────────────────────────
  var SEV_COLORS = {
    green: '#22c55e', yellow: '#eab308',
    orange: '#f97316', red: '#ef4444'
  };
  var RULES = [
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
  var BASE = window.location.origin;

  function apiFetch(path, token, opts) {
    opts = opts || {};
    return fetch(BASE + path, {
      method: opts.method || 'GET',
      headers: Object.assign({
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token,
      }, opts.headers || {}),
      body: opts.body || undefined,
    }).then(function(resp) {
      if (!resp.ok) throw new Error('API ' + resp.status + ': ' + path);
      return resp.json();
    });
  }

  // ─── api() helper (compatible with monolith api(token) pattern) ──
  function api(token) {
    return {
      get: function(path) { return apiFetch(path, token); },
      post: function(path, body) {
        return apiFetch(path, token, {
          method: 'POST',
          body: JSON.stringify(body),
        });
      },
    };
  }

  // ─── Auth Hook ────────────────────────────────────────────
  function useAuth() {
    var tokenState = useState(localStorage.getItem('bhairav_token') || '');
    var token = tokenState[0], setToken = tokenState[1];
    var meState = useState(null);
    var me = meState[0], setMe = meState[1];
    var roleState = useState('');
    var role = roleState[0], setRole = roleState[1];

    useEffect(function() {
      if (!token) return;
      apiFetch('/api/auth/me', token)
        .then(function(u) { setMe(u); setRole(u.role || ''); })
        .catch(function() { setToken(''); localStorage.removeItem('bhairav_token'); });
    }, [token]);

    var login = useCallback(function(user, pass) {
      return fetch(BASE + '/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user, password: pass }),
      }).then(function(r) { return r.json(); }).then(function(d) {
        if (d.token) {
          setToken(d.token);
          localStorage.setItem('bhairav_token', d.token);
          return d;
        }
        throw new Error(d.detail || 'Login failed');
      });
    }, []);

    var logout = useCallback(function() {
      setToken('');
      setMe(null);
      setRole('');
      localStorage.removeItem('bhairav_token');
    }, []);

    return { token: token, me: me, role: role, login: login, logout: logout };
  }

  // ─── Shared WebSocket Manager ─────────────────────────────
  function WSManager() {
    this._connections = {};
    this._listeners = {};
    this._reconnectTimers = {};
  }

  WSManager.prototype.connect = function(endpoint, token, opts) {
    opts = opts || {};
    var onMessage = opts.onMessage;
    var onStatus = opts.onStatus;
    var key = endpoint;
    var self = this;

    if (this._connections[key]) {
      if (onMessage) {
        this._listeners[key] = this._listeners[key] || [];
        this._listeners[key].push(onMessage);
      }
      return this._connections[key];
    }

    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var url = proto + '//' + location.host + endpoint + '?token=' + encodeURIComponent(token);
    var retry = 0;
    var alive = true;

    function doConnect() {
      if (!alive) return;
      var ws = new WebSocket(url);
      self._connections[key] = ws;

      ws.onopen = function() {
        retry = 0;
        if (onStatus) onStatus('connected');
      };

      ws.onmessage = function(e) {
        try {
          var data = JSON.parse(e.data);
          if (onMessage) onMessage(data);
          (self._listeners[key] || []).forEach(function(fn) {
            try { fn(data); } catch(err) { console.error('WS listener error:', err); }
          });
        } catch(err) {}
      };

      ws.onclose = function() {
        if (!alive) return;
        if (onStatus) onStatus('disconnected');
        retry++;
        self._reconnectTimers[key] = setTimeout(doConnect, Math.min(2000 * retry, 15000));
      };

      ws.onerror = function() {};
    }

    doConnect();

    return {
      close: function() {
        alive = false;
        clearTimeout(self._reconnectTimers[key]);
        if (self._connections[key]) {
          self._connections[key].close();
          delete self._connections[key];
        }
      },
      subscribe: function(fn) {
        if (!self._listeners[key]) self._listeners[key] = [];
        self._listeners[key].push(fn);
      },
    };
  };

  WSManager.prototype.disconnect = function(endpoint) {
    var key = endpoint;
    clearTimeout(this._reconnectTimers[key]);
    if (this._connections[key]) {
      this._connections[key].close();
      delete this._connections[key];
    }
    delete this._listeners[key];
  };

  WSManager.prototype.disconnectAll = function() {
    var self = this;
    Object.keys(this._connections).forEach(function(k) { self.disconnect(k); });
  };

  var wsManager = new WSManager();

  // ─── Reusable WebSocket Hook ──────────────────────────────
  function useSharedWS(endpoint, token, onMessage) {
    var statusState = useState('connecting');
    var setStatus = statusState[1];
    var handlerRef = useRef(onMessage);
    handlerRef.current = onMessage;

    useEffect(function() {
      if (!token || !endpoint) return;
      var wrappedHandler = function(data) { handlerRef.current && handlerRef.current(data); };
      var conn = wsManager.connect(endpoint, token, {
        onMessage: wrappedHandler,
        onStatus: setStatus,
      });
      return function() {};
    }, [endpoint, token]);

    return statusState[0];
  }

  // ─── Image Cache ──────────────────────────────────────────
  var snapCache = {};
  var SNAP_CACHE_MAX = 300;

  function cacheSnap(eventId, b64) {
    if (Object.keys(snapCache).length >= SNAP_CACHE_MAX) {
      var oldest = Object.keys(snapCache)[0];
      delete snapCache[oldest];
    }
    snapCache[eventId] = b64;
  }

  function snapUrl(token, eventId) {
    if (snapCache[eventId]) return 'data:image/jpeg;base64,' + snapCache[eventId];
    return '/api/evidence/' + eventId + '/snap?token=' + encodeURIComponent(token);
  }

  function downloadClip(token, ev) {
    return fetch('/api/evidence/' + ev.event_id + '/clip', {
      headers: { 'Authorization': 'Bearer ' + token }
    }).then(function(r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.blob();
    }).then(function(blob) {
      var u = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = u; a.download = ev.event_id + '.mp4';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function() { URL.revokeObjectURL(u); }, 5000);
    }).catch(function() {});
  }

  // ─── SnapImg React component ──────────────────────────────
  function SnapImg(props) {
    var token = props.token;
    var eventId = props.eventId;
    var style = props.style || {};
    var srcState = useState(null);
    var src = srcState[0], setSrc = srcState[1];

    useEffect(function() {
      if (!eventId) return;
      if (snapCache[eventId]) {
        setSrc('data:image/jpeg;base64,' + snapCache[eventId]);
        return;
      }
      fetch('/api/evidence/' + eventId + '/snap', {
        headers: { 'Authorization': 'Bearer ' + token }
      }).then(function(r) { return r.ok ? r.blob() : null; }).then(function(blob) {
        if (!blob) return;
        var reader = new FileReader();
        reader.onload = function() {
          var b64 = String(reader.result).split(',')[1];
          cacheSnap(eventId, b64);
          setSrc('data:image/jpeg;base64,' + b64);
        };
        reader.readAsDataURL(blob);
      }).catch(function() {});
    }, [eventId, token]);

    if (!src) return React.createElement('div', {
      className: 'sighting-thumb',
      style: Object.assign({ height: 150 }, style)
    });
    return React.createElement('img', {
      src: src, alt: 'snapshot',
      style: Object.assign({ width: '100%', height: 150, objectFit: 'cover', display: 'block', background: '#000' }, style)
    });
  }

  // ─── Export to window ─────────────────────────────────────
  window.BHAIRAV = {
    SEV_COLORS: SEV_COLORS, RULES: RULES, BASE: BASE,
    sevStyle: sevStyle,
    apiFetch: apiFetch, api: api,
    useAuth: useAuth,
    wsManager: wsManager, useSharedWS: useSharedWS,
    snapCache: snapCache, cacheSnap: cacheSnap, SNAP_CACHE_MAX: SNAP_CACHE_MAX,
    snapUrl: snapUrl, SnapImg: SnapImg, downloadClip: downloadClip,
    useState: useState, useEffect: useEffect, useRef: useRef, useCallback: useCallback,
  };

})();
