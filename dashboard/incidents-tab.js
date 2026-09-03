/* BHAIRAV City Safety — Incidents Operator Dashboard */
/* Canvas-based map with real-time incident + officer markers */

function IncidentsTab(props) {
  var token = props.token;
  var canvasRef = React.useRef(null);
  var/incidentsState = React.useState([]);
  var incidents = incidentsState[0], setIncidents = incidentsState[1];
  var officersState = React.useState([]);
  var officers = officersState[0], setOfficers = officersState[1];
  var statsState = React.useState({});
  var stats = statsState[0], setStats = statsState[1];
  var filterStatusState = React.useState('');
  var filterStatus = filterStatusState[0], setFilterStatus = filterStatusState[1];
  var filterCategoryState = React.useState('');
  var filterCategory = filterCategoryState[0], setFilterCategory = filterCategoryState[1];
  var selectedState = React.useState(null);
  var selected = selectedState[0], setSelected = selectedState[1];
  var wsState = React.useState(null);
  var ws = wsState[0], setWs = wsState[1];
  var mapState = React.useRef({ offsetX: 0, offsetY: 0, zoom: 1.0, dragging: false, lastX: 0, lastY: 0 });

  var apiFn = (window.BHAIRAV || {}).api || function() { return { get: function() { return Promise.resolve({}); }, post: function() { return Promise.resolve({}); } }; };

  var CATEGORY_EMOJI = { medical: '🚑', fire: '🔥', crime: '⚠️', road_accident: '🚗', disaster: '🌊', missing_person: '👤', other: '📋' };
  var STATUS_COLORS = { reported: '#eab308', verified: '#f97316', dispatched: '#22d3ee', en_route: '#3b82f6', on_scene: '#8b5cf6', resolved: '#22c55e', cancelled: '#64748b' };
  var LEVEL_COLORS = { 1: '#22c55e', 2: '#eab308', 3: '#f97316', 4: '#ef4444' };
  var OFFICER_STATUS_COLORS = { available: '#22c55e', dispatched: '#f97316', en_route: '#3b82f6', on_scene: '#8b5cf6', off_duty: '#64748b' };

  // Fetch data
  function fetchData() {
    apiFn(token).get('/api/incidents?limit=200').then(function(d) {
      setIncidents(d.incidents || []);
    }).catch(function() {});
    apiFn(token).get('/api/officers').then(function(d) {
      setOfficers(d.officers || []);
    }).catch(function() {});
    apiFn(token).get('/api/incidents/stats').then(function(d) {
      setStats(d);
    }).catch(function() {});
  }

  React.useEffect(function() { fetchData(); }, [token]);

  // WebSocket for real-time updates
  React.useEffect(function() {
    var proto = location.protocol === 'https:' ? 'wss' : 'ws';
    var sock = new WebSocket(proto + '://' + location.host + '/ws/incidents?token=' + token);
    sock.onmessage = function(ev) {
      try {
        var msg = JSON.parse(ev.data);
        if (msg.type === 'incident') {
          // Update or add incident
          var inc = msg.incident;
          setIncidents(function(prev) {
            var idx = prev.findIndex(function(i) { return i.id === inc.id; });
            if (idx >= 0) { var next = prev.slice(); next[idx] = inc; return next; }
            return [inc].concat(prev);
          });
        } else if (msg.type === 'snapshot') {
          setStats(msg.data || {});
        }
      } catch (e) {}
    };
    sock.onclose = function() { setTimeout(function() { fetchData(); }, 2000); };
    setWs(sock);
    return function() { if (sock) sock.close(); };
  }, [token]);

  // Filter incidents
  var filtered = incidents.filter(function(inc) {
    if (filterStatus && inc.status !== filterStatus) return false;
    if (filterCategory && inc.category !== filterCategory) return false;
    return true;
  });

  // Canvas map rendering
  React.useEffect(function() {
    var canvas = canvasRef.current;
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var W = canvas.clientWidth;
    var H = canvas.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    var ms = mapState.current;
    ctx.fillStyle = '#0a0f1a';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = 'rgba(56,189,248,0.05)';
    ctx.lineWidth = 1;
    var gridSize = 30 * ms.zoom;
    var offX = ms.offsetX % gridSize;
    var offY = ms.offsetY % gridSize;
    for (var x = offX; x < W; x += gridSize) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (var y = offY; y < H; y += gridSize) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    // Calculate bounds from data
    var allLats = [], allLngs = [];
    incidents.forEach(function(i) { allLats.push(i.location.lat); allLngs.push(i.location.lng); });
    officers.forEach(function(o) { allLats.push(o.location.lat); allLngs.push(o.location.lng); });
    if (allLats.length === 0) { allLats = [28.6139]; allLngs = [77.2090]; }
    var minLat = Math.min.apply(null, allLats), maxLat = Math.max.apply(null, allLats);
    var minLng = Math.min.apply(null, allLngs), maxLng = Math.max.apply(null, allLngs);
    var latRange = Math.max(maxLat - minLat, 0.005) * 1.3;
    var lngRange = Math.max(maxLng - minLng, 0.005) * 1.3;
    var centerLat = (minLat + maxLat) / 2;
    var centerLng = (minLng + maxLng) / 2;

    function toScreen(lat, lng) {
      var x = ((lng - centerLng) / lngRange + 0.5) * W * ms.zoom + ms.offsetX + W / 2 * (1 - ms.zoom);
      var y = ((centerLat - lat) / latRange + 0.5) * H * ms.zoom + ms.offsetY + H / 2 * (1 - ms.zoom);
      return [x, y];
    }

    // Draw officer markers (triangles)
    officers.forEach(function(off) {
      var pos = toScreen(off.location.lat, off.location.lng);
      var x = pos[0], y = pos[1];
      if (x < -20 || x > W + 20 || y < -20 || y > H + 20) return;
      var col = OFFICER_STATUS_COLORS[off.status] || '#64748b';
      ctx.beginPath();
      ctx.moveTo(x, y - 10);
      ctx.lineTo(x - 7, y + 5);
      ctx.lineTo(x + 7, y + 5);
      ctx.closePath();
      ctx.fillStyle = col + '88';
      ctx.fill();
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // Label
      ctx.fillStyle = col;
      ctx.font = '9px system-ui';
      ctx.fillText(off.name.split(' ')[0], x + 10, y + 3);
    });

    // Draw incident markers (circles with pulse for critical)
    filtered.forEach(function(inc) {
      var pos = toScreen(inc.location.lat, inc.location.lng);
      var x = pos[0], y = pos[1];
      if (x < -20 || x > W + 20 || y < -20 || y > H + 20) return;
      var col = LEVEL_COLORS[inc.emergency_level] || '#eab308';
      var radius = 6 + inc.emergency_level * 3;
      var isSelected = selected && selected.id === inc.id;

      // Pulse for critical
      if (inc.emergency_level >= 4 && inc.status !== 'resolved') {
        var pulse = (Date.now() % 2000) / 2000;
        ctx.beginPath();
        ctx.arc(x, y, radius + pulse * 15, 0, Math.PI * 2);
        ctx.fillStyle = col + Math.round((1 - pulse) * 40).toString(16).padStart(2, '0');
        ctx.fill();
      }

      // Marker circle
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
      if (isSelected) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Icon text
      ctx.fillStyle = '#fff';
      ctx.font = (radius - 2) + 'px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(CATEGORY_EMOJI[inc.category] || '?', x, y);
      ctx.textAlign = 'start';
      ctx.textBaseline = 'alphabetic';

      // Status badge
      ctx.fillStyle = STATUS_COLORS[inc.status] || '#64748b';
      ctx.font = 'bold 9px system-ui';
      ctx.fillText(inc.status.toUpperCase(), x + radius + 4, y + 3);
    });

    // Scale indicator
    ctx.fillStyle = '#475569';
    ctx.font = '11px system-ui';
    ctx.fillText(filtered.length + ' incidents · ' + officers.length + ' officers', 12, H - 12);
  }, [filtered, officers, selected, incidents]);

  // Canvas pan/zoom
  function onCanvasMouseDown(e) {
    var ms = mapState.current;
    ms.dragging = true;
    ms.lastX = e.clientX;
    ms.lastY = e.clientY;
  }
  function onCanvasMouseMove(e) {
    var ms = mapState.current;
    if (!ms.dragging) return;
    ms.offsetX += e.clientX - ms.lastX;
    ms.offsetY += e.clientY - ms.lastY;
    ms.lastX = e.clientX;
    ms.lastY = e.clientY;
  }
  function onCanvasMouseUp() { mapState.current.dragging = false; }
  function onCanvasWheel(e) {
    e.preventDefault();
    var ms = mapState.current;
    var factor = e.deltaY > 0 ? 0.9 : 1.1;
    ms.zoom = Math.max(0.3, Math.min(5, ms.zoom * factor));
  }
  function onCanvasClick(e) {
    var canvas = canvasRef.current;
    var rect = canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;
    // Find closest incident
    var best = null, bestDist = 20;
    filtered.forEach(function(inc) {
      var pos = (function() {
        var allLats = [], allLngs = [];
        incidents.forEach(function(i) { allLats.push(i.location.lat); allLngs.push(i.location.lng); });
        officers.forEach(function(o) { allLats.push(o.location.lat); allLngs.push(o.location.lng); });
        if (allLats.length === 0) { allLats = [28.6139]; allLngs = [77.2090]; }
        var minLat = Math.min.apply(null, allLats), maxLat = Math.max.apply(null, allLats);
        var minLng = Math.min.apply(null, allLngs), maxLng = Math.max.apply(null, allLngs);
        var latRange = Math.max(maxLat - minLat, 0.005) * 1.3;
        var lngRange = Math.max(maxLng - minLng, 0.005) * 1.3;
        var centerLat = (minLat + maxLat) / 2;
        var centerLng = (minLng + maxLng) / 2;
        var ms = mapState.current;
        var x = ((inc.location.lng - centerLng) / lngRange + 0.5) * canvas.clientWidth * ms.zoom + ms.offsetX + canvas.clientWidth / 2 * (1 - ms.zoom);
        var y = ((centerLat - inc.location.lat) / latRange + 0.5) * canvas.clientHeight * ms.zoom + ms.offsetY + canvas.clientHeight / 2 * (1 - ms.zoom);
        return [x, y];
      })();
      var d = Math.sqrt(Math.pow(pos[0] - mx, 2) + Math.pow(pos[1] - my, 2));
      if (d < bestDist) { bestDist = d; best = inc; }
    });
    setSelected(best);
  }

  // Dispatch action
  function handleDispatch(incidentId) {
    apiFn(token).post('/api/incidents/' + incidentId + '/status', { status: 'dispatched', note: 'Operator dispatched' }).then(function(d) {
      setIncidents(function(prev) { return prev.map(function(i) { return i.id === incidentId ? d.incident : i; }); });
      if (selected && selected.id === incidentId) setSelected(d.incident);
    }).catch(function() {});
  }
  function handleStatusUpdate(incidentId, newStatus) {
    apiFn(token).post('/api/incidents/' + incidentId + '/status', { status: newStatus, note: 'Status updated by operator' }).then(function(d) {
      setIncidents(function(prev) { return prev.map(function(i) { return i.id === incidentId ? d.incident : i; }); });
      if (selected && selected.id === incidentId) setSelected(d.incident);
    }).catch(function() {});
  }

  // Stats cards
  var activeIncidents = stats.active_incidents || incidents.filter(function(i) { return i.status !== 'resolved' && i.status !== 'cancelled'; }).length;
  var availableOfficers = stats.available_officers || officers.filter(function(o) { return o.status === 'available'; }).length;
  var criticalCount = incidents.filter(function(i) { return i.emergency_level >= 4 && i.status !== 'resolved'; }).length;

  return React.createElement('div', { style: { display: 'flex', gap: 16, height: 'calc(100vh - 120px)' } },
    // Left: Map + Filters
    React.createElement('div', { style: { flex: 1, display: 'flex', flexDirection: 'column', gap: 12 } },
      // Stats bar
      React.createElement('div', { style: { display: 'flex', gap: 12 } },
        React.createElement('div', { className: 'pill', style: { background: activeIncidents > 0 ? 'rgba(234,179,8,0.1)' : 'transparent' } },
          React.createElement('span', { className: 'dot ' + (activeIncidents > 0 ? 'live' : 'idle') }),
          ' ' + activeIncidents + ' active'
        ),
        React.createElement('div', { className: 'pill' }, availableOfficers + ' available'),
        criticalCount > 0 ? React.createElement('div', { className: 'pill', style: { color: '#ef4444', borderColor: '#ef4444' } }, '🚨 ' + criticalCount + ' CRITICAL') : null,
        React.createElement('div', { className: 'spacer' }),
        React.createElement('select', {
          value: filterStatus,
          onChange: function(e) { setFilterStatus(e.target.value); },
          style: { background: '#0d1526', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 8, padding: '5px 8px', fontSize: 12 }
        },
          React.createElement('option', { value: '' }, 'All Status'),
          ['reported', 'verified', 'dispatched', 'en_route', 'on_scene', 'resolved', 'cancelled'].map(function(s) {
            return React.createElement('option', { key: s, value: s }, s.replace('_', ' '));
          })
        ),
        React.createElement('select', {
          value: filterCategory,
          onChange: function(e) { setFilterCategory(e.target.value); },
          style: { background: '#0d1526', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 8, padding: '5px 8px', fontSize: 12 }
        },
          React.createElement('option', { value: '' }, 'All Categories'),
          ['medical', 'fire', 'crime', 'road_accident', 'disaster', 'missing_person', 'other'].map(function(c) {
            return React.createElement('option', { key: c, value: c }, (CATEGORY_EMOJI[c] || '') + ' ' + c.replace('_', ' '));
          })
        )
      ),
      // Canvas Map
      React.createElement('div', { style: { flex: 1, background: '#0a0f1a', border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden', position: 'relative' } },
        React.createElement('canvas', {
          ref: canvasRef,
          style: { width: '100%', height: '100%', cursor: 'grab' },
          onMouseDown: onCanvasMouseDown,
          onMouseMove: onCanvasMouseMove,
          onMouseUp: onCanvasMouseUp,
          onMouseLeave: onCanvasMouseUp,
          onWheel: onCanvasWheel,
          onClick: onCanvasClick
        }),
        React.createElement('div', { style: { position: 'absolute', bottom: 10, left: 10, background: 'rgba(15,23,42,0.85)', padding: '6px 10px', borderRadius: 8, fontSize: 11, color: '#94a3b8' } },
          '🖱 drag to pan · scroll to zoom · click incident to inspect'
        )
      )
    ),
    // Right: Incident List + Detail Panel
    React.createElement('div', { style: { width: 380, display: 'flex', flexDirection: 'column', gap: 12 } },
      // Detail Panel (when selected)
      selected ? React.createElement('div', { style: { background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 12, padding: 14 } },
        React.createElement('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 } },
          React.createElement('span', { style: { fontWeight: 700, fontSize: 14 } },
            CATEGORY_EMOJI[selected.category] + ' ' + selected.id.slice(0, 8)
          ),
          React.createElement('button', {
            onClick: function() { setSelected(null); },
            style: { background: 'none', border: 'none', color: 'var(--muted)', fontSize: 16, cursor: 'pointer' }
          }, '✕')
        ),
        React.createElement('div', { style: { display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' } },
          React.createElement('span', { className: 'sev-badge', style: { background: (LEVEL_COLORS[selected.emergency_level] || '#eab308') + '22', color: LEVEL_COLORS[selected.emergency_level] || '#eab308', border: '1px solid ' + (LEVEL_COLORS[selected.emergency_level] || '#eab308') + '66' } },
            'LVL ' + selected.emergency_level
          ),
          React.createElement('span', { className: 'sev-badge', style: { background: (STATUS_COLORS[selected.status] || '#64748b') + '22', color: STATUS_COLORS[selected.status] || '#64748b', border: '1px solid ' + (STATUS_COLORS[selected.status] || '#64748b') + '66' } },
            selected.status.toUpperCase()
          ),
          selected.source ? React.createElement('span', { className: 'sev-badge', style: { background: '#1e293b', color: '#94a3b8' } }, selected.source) : null,
          selected.crowd_reports > 1 ? React.createElement('span', { className: 'sev-badge', style: { background: 'rgba(234,179,8,0.15)', color: '#eab308' } }, '👥 x' + selected.crowd_reports) : null
        ),
        React.createElement('div', { style: { fontSize: 13, color: 'var(--text)', marginBottom: 6 } }, selected.description || 'No description'),
        React.createElement('div', { style: { fontSize: 11, color: 'var(--muted)', marginBottom: 4 } },
          '📍 ' + (selected.location_name || '') + ' (' + selected.location.lat.toFixed(4) + ', ' + selected.location.lng.toFixed(4) + ')'
        ),
        React.createElement('div', { style: { fontSize: 11, color: 'var(--muted)', marginBottom: 10 } },
          '👤 ' + (selected.reporter_name || 'Anonymous') + ' · ' + new Date(selected.created_at * 1000).toLocaleString()
        ),
        // Assigned officers
        selected.assigned_officers && selected.assigned_officers.length > 0 ? React.createElement('div', { style: { marginBottom: 10 } },
          React.createElement('div', { style: { fontSize: 11, color: 'var(--muted)', marginBottom: 4 } }, 'Assigned Officers:'),
          selected.assigned_officers.map(function(oid) {
            var off = officers.find(function(o) { return o.id === oid; });
            return React.createElement('div', { key: oid, style: { fontSize: 12, color: 'var(--text)', padding: '4px 0' } },
              off ? (off.name + ' (' + off.role + ')') : oid
            );
          })
        ) : null,
        // Action buttons
        React.createElement('div', { style: { display: 'flex', gap: 8, flexWrap: 'wrap' } },
          selected.status === 'reported' || selected.status === 'verified' ? React.createElement('button', {
            className: 'btn',
            style: { background: 'var(--cyan)', color: '#06222b', fontSize: 12, padding: '6px 12px' },
            onClick: function() { handleDispatch(selected.id); }
          }, '📟 Dispatch') : null,
          selected.status === 'dispatched' ? React.createElement('button', {
            className: 'btn',
            style: { background: '#3b82f6', color: '#fff', fontSize: 12, padding: '6px 12px' },
            onClick: function() { handleStatusUpdate(selected.id, 'en_route'); }
          }, '🚀 En Route') : null,
          selected.status === 'en_route' ? React.createElement('button', {
            className: 'btn',
            style: { background: '#8b5cf6', color: '#fff', fontSize: 12, padding: '6px 12px' },
            onClick: function() { handleStatusUpdate(selected.id, 'on_scene'); }
          }, '📍 On Scene') : null,
          selected.status === 'on_scene' ? React.createElement('button', {
            className: 'btn',
            style: { background: 'var(--green)', color: '#fff', fontSize: 12, padding: '6px 12px' },
            onClick: function() { handleStatusUpdate(selected.id, 'resolved'); }
          }, '✅ Resolve') : null,
          selected.status !== 'resolved' && selected.status !== 'cancelled' ? React.createElement('button', {
            className: 'btn',
            style: { background: '#1e293b', color: 'var(--muted)', fontSize: 12, padding: '6px 12px' },
            onClick: function() { handleStatusUpdate(selected.id, 'cancelled'); }
          }, '✖ Cancel') : null
        ),
        // Timeline
        selected.timeline && selected.timeline.length > 0 ? React.createElement('div', { style: { marginTop: 10 } },
          React.createElement('div', { style: { fontSize: 11, color: 'var(--muted)', marginBottom: 4 } }, 'Timeline:'),
          selected.timeline.slice(-5).reverse().map(function(ev, i) {
            return React.createElement('div', { key: i, style: { fontSize: 11, padding: '3px 0', borderBottom: '1px solid var(--border2)' } },
              React.createElement('span', { style: { color: STATUS_COLORS[ev.status] || '#94a3b8', fontWeight: 600 } }, ev.status),
              ' ',
              React.createElement('span', { style: { color: 'var(--muted)' } }, new Date(ev.time * 1000).toLocaleTimeString()),
              ev.note ? React.createElement('span', { style: { color: 'var(--dim)' } }, ' — ' + ev.note) : null
            );
          })
        ) : null
      ) : null,
      // Incident list
      React.createElement('div', { className: 'feed', style: { flex: 1, minHeight: 0 } },
        React.createElement('div', { className: 'feed-head', style: { display: 'flex', justifyContent: 'space-between' } },
          React.createElement('span', null, '🚨 Incidents (' + filtered.length + ')'),
          React.createElement('button', {
            onClick: fetchData,
            style: { background: 'none', border: '1px solid var(--border)', color: 'var(--muted)', borderRadius: 6, padding: '2px 8px', fontSize: 11, cursor: 'pointer' }
          }, '🔄 Refresh')
        ),
        React.createElement('div', { className: 'feed-list' },
          filtered.length === 0 ? React.createElement('div', { className: 'feed-empty' }, 'No incidents match filters') :
          filtered.map(function(inc) {
            return React.createElement('div', {
              key: inc.id,
              onClick: function() { setSelected(inc); },
              style: {
                display: 'flex', alignItems: 'center', gap: 10, padding: '9px 14px',
                borderBottom: '1px solid var(--border2)', fontSize: 12, cursor: 'pointer',
                background: selected && selected.id === inc.id ? '#0d1526' : 'transparent',
                transition: 'background .15s'
              }
            },
              React.createElement('span', { className: 'sev', style: { background: LEVEL_COLORS[inc.emergency_level] || '#eab308' } }),
              React.createElement('div', { style: { flex: 1 } },
                React.createElement('div', { style: { color: 'var(--text)', fontWeight: 600 } },
                  (CATEGORY_EMOJI[inc.category] || '?') + ' ' + (inc.location_name || inc.id.slice(0, 8))
                ),
                React.createElement('div', { style: { color: 'var(--muted)', fontSize: 11 } },
                  inc.description ? inc.description.slice(0, 50) : 'No details'
                )
              ),
              React.createElement('div', { style: { textAlign: 'right' } },
                React.createElement('div', { style: { color: STATUS_COLORS[inc.status] || '#94a3b8', fontSize: 10, fontWeight: 700 } },
                  inc.status.toUpperCase()
                ),
                React.createElement('div', { style: { color: 'var(--muted)', fontSize: 10 } },
                  inc.emergency_level >= 4 ? '🚨 CRITICAL' : ('LVL ' + inc.emergency_level)
                )
              )
            );
          })
        )
      )
    )
  );
}
