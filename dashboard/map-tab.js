/* Phase 16: Interactive camera map with FOV cones, re-ID trails, and heatmap overlay. */
/* Converted from JSX to React.createElement for direct script loading. */

function MapTab(props) {
  var token = props.token;
  var canvasRef = React.useRef(null);
  var camerasState = React.useState([]);
  var cameras = camerasState[0], setCameras = camerasState[1];
  var subjectsState = React.useState([]);
  var subjects = subjectsState[0], setSubjects = subjectsState[1];
  var heatmapState = React.useState(null);
  var heatmap = heatmapState[0], setHeatmap = heatmapState[1];
  var selectedCamState = React.useState(null);
  var selectedCam = selectedCamState[0], setSelectedCam = selectedCamState[1];
  var showHeatmapState = React.useState(true);
  var showHeatmap = showHeatmapState[0], setShowHeatmap = showHeatmapState[1];
  var showTrailsState = React.useState(true);
  var showTrails = showTrailsState[0], setShowTrails = showTrailsState[1];

  var CAM_POSITIONS = React.useRef({
    'CAM-01': { x: 0.2, y: 0.3, fov: 45, rot: 30, label: 'Plaza NW' },
    'CAM-02': { x: 0.5, y: 0.2, fov: 60, rot: 90, label: 'Plaza NE' },
    'CAM-03': { x: 0.8, y: 0.3, fov: 45, rot: 150, label: 'Server Entry' },
    'CAM-04': { x: 0.5, y: 0.7, fov: 60, rot: 270, label: 'Parking S' },
    'CAM-05': { x: 0.2, y: 0.8, fov: 45, rot: 330, label: 'Gate SW' },
    'CAM-06': { x: 0.8, y: 0.8, fov: 45, rot: 30, label: 'Gate SE' },
  });

  var apiFn = (window.BHAIRAV || {}).api || function() { return { get: function() { return Promise.resolve({}); } }; };

  React.useEffect(function() {
    function load() {
      apiFn(token).get('/api/reid/subjects').then(function(data) {
        setSubjects(data.subjects || []);
      }).catch(function() {});
      apiFn(token).get('/api/status').then(function(data) {
        if (data.cameras) setCameras(data.cameras);
      }).catch(function() {});
    }
    load();
    var iv = setInterval(load, 5000);
    return function() { clearInterval(iv); };
  }, [token]);

  React.useEffect(function() {
    var ws;
    try {
      var base = location.origin || 'http://localhost:8000';
      ws = new WebSocket(base.replace('http', 'ws') + '/ws/analytics?token=' + token);
      ws.onmessage = function(ev) {
        try {
          var m = JSON.parse(ev.data);
          if (m.type === 'analytics' && m.data && m.data.heatmap) {
            setHeatmap(m.data.heatmap);
          }
        } catch (e) {}
      };
    } catch (e) {}
    return function() { if (ws) ws.close(); };
  }, [token]);

  React.useEffect(function() {
    var canvas = canvasRef.current;
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var W = canvas.width;
    var H = canvas.height;
    var cams = CAM_POSITIONS.current;

    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = 'rgba(56,189,248,0.06)';
    ctx.lineWidth = 1;
    for (var i = 0; i < W; i += 40) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, H); ctx.stroke();
    }
    for (var j = 0; j < H; j += 40) {
      ctx.beginPath(); ctx.moveTo(0, j); ctx.lineTo(W, j); ctx.stroke();
    }

    // Heatmap
    if (showHeatmap && heatmap && heatmap.grid) {
      var grid = heatmap.grid;
      var gw = (grid[0] || []).length || 32;
      var gh = grid.length || 24;
      var cellW = W / gw;
      var cellH = H / gh;
      for (var r = 0; r < gh; r++) {
        for (var c = 0; c < gw; c++) {
          var v = (grid[r] || [])[c] || 0;
          if (v > 0.01) {
            var alpha = Math.min(v * 0.7, 0.5);
            ctx.fillStyle = 'rgba(239,68,68,' + alpha + ')';
            ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
          }
        }
      }
    }

    // FOV cones
    var camIds = Object.keys(cams);
    for (var ci = 0; ci < camIds.length; ci++) {
      var id = camIds[ci];
      var cam = cams[id];
      var cx = cam.x * W;
      var cy = cam.y * H;
      var fovRad = (cam.fov * Math.PI) / 180;
      var rotRad = (cam.rot * Math.PI) / 180;
      var radius = Math.min(W, H) * 0.22;
      var isSelected = selectedCam === id;
      var isActive = cameras.some(function(c) { return c.id === id || c === id; });

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, radius, rotRad - fovRad / 2, rotRad + fovRad / 2);
      ctx.closePath();
      ctx.fillStyle = isSelected ? 'rgba(34,211,238,0.25)' : isActive ? 'rgba(34,197,94,0.15)' : 'rgba(148,163,184,0.08)';
      ctx.fill();
      ctx.strokeStyle = isSelected ? '#22d3ee' : isActive ? '#22c55e' : '#475569';
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = isActive ? '#22c55e' : '#64748b';
      ctx.fill();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(cam.label, cx, cy + 20);
      ctx.fillStyle = '#64748b';
      ctx.font = '10px system-ui, sans-serif';
      ctx.fillText(id, cx, cy + 32);
    }

    // Re-ID trails
    if (showTrails && subjects.length > 0) {
      var trailColors = ['#a855f7', '#f59e0b', '#06b6d4', '#ef4444', '#22c55e'];
      var trailSubjects = subjects.slice(0, 5);
      for (var si = 0; si < trailSubjects.length; si++) {
        var subj = trailSubjects[si];
        var cams_seen = subj.cameras || [];
        if (cams_seen.length < 2) continue;
        var color = trailColors[si % trailColors.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        var started = false;
        for (var ti = 0; ti < cams_seen.length; ti++) {
          var cam2 = cams[cams_seen[ti]];
          if (!cam2) continue;
          var px = cam2.x * W;
          var py = cam2.y * H;
          if (!started) { ctx.moveTo(px, py); started = true; }
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        if (started) {
          var firstCam = cams[cams_seen[0]];
          if (firstCam) {
            ctx.fillStyle = color;
            ctx.font = 'bold 10px system-ui, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(subj.name || subj.id, firstCam.x * W + 12, firstCam.y * H - 8);
          }
        }
      }
    }

    // Legend
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(8, H - 60, 180, 52);
    ctx.strokeStyle = '#334155';
    ctx.strokeRect(8, H - 60, 180, 52);
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#22c55e'; ctx.fillRect(16, H - 48, 8, 8);
    ctx.fillStyle = '#94a3b8'; ctx.fillText('Active camera', 30, H - 40);
    ctx.fillStyle = '#ef4444'; ctx.fillRect(16, H - 32, 8, 8);
    ctx.fillStyle = '#94a3b8'; ctx.fillText('Crowd heatmap', 30, H - 24);
    ctx.fillStyle = '#a855f7'; ctx.fillRect(100, H - 48, 8, 8);
    ctx.fillStyle = '#94a3b8'; ctx.fillText('Re-ID trail', 114, H - 40);
    ctx.fillStyle = '#64748b'; ctx.fillRect(100, H - 32, 8, 8);
    ctx.fillStyle = '#94a3b8'; ctx.fillText('FOV cone', 114, H - 24);
  }, [cameras, subjects, heatmap, selectedCam, showHeatmap, showTrails]);

  function handleClick(e) {
    var canvas = canvasRef.current;
    var rect = canvas.getBoundingClientRect();
    var mx = (e.clientX - rect.left) / rect.width;
    var my = (e.clientY - rect.top) / rect.height;
    var cams = CAM_POSITIONS.current;
    for (var id in cams) {
      var cam = cams[id];
      var dx = mx - cam.x;
      var dy = my - cam.y;
      if (Math.sqrt(dx * dx + dy * dy) < 0.04) {
        setSelectedCam(selectedCam === id ? null : id);
        return;
      }
    }
    setSelectedCam(null);
  }

  var camSubs = selectedCam ? subjects.filter(function(s) { return (s.cameras || []).indexOf(selectedCam) >= 0; }) : [];

  return React.createElement('div', {className: 'card'},
    React.createElement('h2', null, '\u{1f5fa} Site Map ', React.createElement('span', {className: 'muted'}, '\u00b7 camera positions, FOV, re-ID trails, heatmap')),
    React.createElement('div', {className: 'chips', style: {marginBottom: 12}},
      React.createElement('button', {
        className: 'chip',
        style: {cursor:'pointer', background: showHeatmap ? '#422006' : '#1e293b', color: showHeatmap ? '#fbbf24' : 'var(--muted)', border: 'none', padding: '4px 12px', borderRadius: 6},
        onClick: function() { setShowHeatmap(!showHeatmap); }
      }, '\u{1f525} Heatmap ' + (showHeatmap ? 'ON' : 'OFF')),
      React.createElement('button', {
        className: 'chip',
        style: {cursor:'pointer', background: showTrails ? '#1e1b4b' : '#1e293b', color: showTrails ? '#a5b4fc' : 'var(--muted)', border: 'none', padding: '4px 12px', borderRadius: 6},
        onClick: function() { setShowTrails(!showTrails); }
      }, '\u{1f464} Trails ' + (showTrails ? 'ON' : 'OFF')),
      React.createElement('span', {className: 'chip', style: {background:'#052e16', color:'var(--green)'}},
        (cameras.length || Object.keys(CAM_POSITIONS.current).length) + ' cameras'),
      React.createElement('span', {className: 'chip', style: {background:'#1e1b4b', color:'#a5b4fc'}},
        subjects.length + ' re-ID subjects')
    ),
    React.createElement('canvas', {
      ref: canvasRef, width: 800, height: 500,
      style: {width:'100%', borderRadius:8, border:'1px solid #1e293b', cursor:'pointer'},
      onClick: handleClick
    }),
    selectedCam && React.createElement('div', {style: {marginTop: 12, padding: '8px 12px', background: '#0d1526', borderRadius: 8, fontSize: 13}},
      React.createElement('b', {style: {color:'var(--cyan)'}}, selectedCam),
      React.createElement('span', {className: 'muted', style: {marginLeft: 8}},
        (CAM_POSITIONS.current[selectedCam] ? CAM_POSITIONS.current[selectedCam].label : '') + ' \u00b7 FOV ' + (CAM_POSITIONS.current[selectedCam] ? CAM_POSITIONS.current[selectedCam].fov : '') + '\u00b0'),
      camSubs.length > 0 ? React.createElement('div', {style: {marginTop: 6, fontSize: 12}},
        React.createElement('span', {className: 'muted'}, 'People seen here: '),
        camSubs.map(function(s) {
          return React.createElement('span', {key: s.id, className: 'chip', style: {background:'#1e1b4b', color:'#a5b4fc', marginRight:4, fontSize:11}},
            s.name || s.id);
        })
      ) : React.createElement('div', {className: 'muted', style: {marginTop:4, fontSize:12}}, 'No re-ID subjects seen yet')
    )
  );
}
