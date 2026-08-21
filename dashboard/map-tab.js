/* Phase 16: Interactive camera map with FOV cones, re-ID trails, and heatmap overlay. */

function MapTab({ token }) {
  const canvasRef = React.useRef(null);
  const [cameras, setCameras] = React.useState([]);
  const [subjects, setSubjects] = React.useState([]);
  const [heatmap, setHeatmap] = React.useState(null);
  const [selectedCam, setSelectedCam] = React.useState(null);
  const [showHeatmap, setShowHeatmap] = React.useState(true);
  const [showTrails, setShowTrails] = React.useState(true);

  const CAM_POSITIONS = React.useRef({
    'CAM-01': { x: 0.2, y: 0.3, fov: 45, rot: 30, label: 'Plaza NW' },
    'CAM-02': { x: 0.5, y: 0.2, fov: 60, rot: 90, label: 'Plaza NE' },
    'CAM-03': { x: 0.8, y: 0.3, fov: 45, rot: 150, label: 'Server Entry' },
    'CAM-04': { x: 0.5, y: 0.7, fov: 60, rot: 270, label: 'Parking S' },
    'CAM-05': { x: 0.2, y: 0.8, fov: 45, rot: 330, label: 'Gate SW' },
    'CAM-06': { x: 0.8, y: 0.8, fov: 45, rot: 30, label: 'Gate SE' },
  });

  React.useEffect(() => {
    async function load() {
      try {
        const data = await api(token).get('/api/reid/subjects');
        setSubjects(data.subjects || []);
      } catch (e) { /* reid might be disabled */ }
      try {
        const data = await api(token).get('/api/status');
        if (data.cameras) setCameras(data.cameras);
      } catch (e) { /* */ }
    }
    load();
    const iv = setInterval(load, 5000);
    return () => clearInterval(iv);
  }, [token]);

  React.useEffect(() => {
    let ws;
    try {
      const base = location.origin || 'http://localhost:8000';
      ws = new WebSocket(base.replace('http', 'ws') + '/ws/analytics?token=' + token);
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          if (m.type === 'analytics' && m.data && m.data.heatmap) {
            setHeatmap(m.data.heatmap);
          }
        } catch (e) { /* */ }
      };
    } catch (e) { /* */ }
    return () => { if (ws) ws.close(); };
  }, [token]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const cams = CAM_POSITIONS.current;

    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = 'rgba(56,189,248,0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i < W; i += 40) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, H); ctx.stroke();
    }
    for (let j = 0; j < H; j += 40) {
      ctx.beginPath(); ctx.moveTo(0, j); ctx.lineTo(W, j); ctx.stroke();
    }

    // Heatmap
    if (showHeatmap && heatmap && heatmap.grid) {
      const grid = heatmap.grid;
      const gw = (grid[0] || []).length || 32;
      const gh = grid.length || 24;
      const cellW = W / gw;
      const cellH = H / gh;
      for (let r = 0; r < gh; r++) {
        for (let c = 0; c < gw; c++) {
          const v = (grid[r] || [])[c] || 0;
          if (v > 0.01) {
            const alpha = Math.min(v * 0.7, 0.5);
            ctx.fillStyle = 'rgba(239,68,68,' + alpha + ')';
            ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
          }
        }
      }
    }

    // FOV cones
    for (const [id, cam] of Object.entries(cams)) {
      const cx = cam.x * W;
      const cy = cam.y * H;
      const fovRad = (cam.fov * Math.PI) / 180;
      const rotRad = (cam.rot * Math.PI) / 180;
      const radius = Math.min(W, H) * 0.22;
      const isSelected = selectedCam === id;
      const isActive = cameras.some(function(c) { return c.id === id || c === id; });

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
      subjects.slice(0, 5).forEach(function(subj, si) {
        var cams_seen = subj.cameras || [];
        if (cams_seen.length < 2) return;
        var color = trailColors[si % trailColors.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        var started = false;
        for (var ci = 0; ci < cams_seen.length; ci++) {
          var cam2 = cams[cams_seen[ci]];
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
      });
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

  return (
    <div className="card">
      <h2>{'\u{1f5fa}'} Site Map <span className="muted">{'\u00b7'} camera positions, FOV, re-ID trails, heatmap</span></h2>
      <div className="chips" style={{marginBottom: 12}}>
        <button className={'chip' + (showHeatmap ? ' active' : '')}
                style={{cursor:'pointer', background: showHeatmap ? '#422006' : '#1e293b',
                        color: showHeatmap ? '#fbbf24' : 'var(--muted)', border: 'none', padding: '4px 12px', borderRadius: 6}}
                onClick={function() { setShowHeatmap(!showHeatmap); }}>
          {'\u{1f525}'} Heatmap {showHeatmap ? 'ON' : 'OFF'}
        </button>
        <button className={'chip' + (showTrails ? ' active' : '')}
                style={{cursor:'pointer', background: showTrails ? '#1e1b4b' : '#1e293b',
                        color: showTrails ? '#a5b4fc' : 'var(--muted)', border: 'none', padding: '4px 12px', borderRadius: 6}}
                onClick={function() { setShowTrails(!showTrails); }}>
          {'\u{1f464}'} Trails {showTrails ? 'ON' : 'OFF'}
        </button>
        <span className="chip" style={{background:'#052e16', color:'var(--green)'}}>
          {cameras.length || Object.keys(CAM_POSITIONS.current).length} cameras
        </span>
        <span className="chip" style={{background:'#1e1b4b', color:'#a5b4fc'}}>
          {subjects.length} re-ID subjects
        </span>
      </div>
      <canvas ref={canvasRef} width={800} height={500}
              style={{width:'100%', borderRadius:8, border:'1px solid #1e293b', cursor:'pointer'}}
              onClick={handleClick} />
      {selectedCam && (
        <div style={{marginTop: 12, padding: '8px 12px', background: '#0d1526', borderRadius: 8, fontSize: 13}}>
          <b style={{color:'var(--cyan)'}}>{selectedCam}</b>
          <span className="muted" style={{marginLeft: 8}}>
            {CAM_POSITIONS.current[selectedCam] ? CAM_POSITIONS.current[selectedCam].label : ''}{' \u00b7 FOV '}
            {CAM_POSITIONS.current[selectedCam] ? CAM_POSITIONS.current[selectedCam].fov : ''}{'\u00b0'}
          </span>
          {camSubs.length > 0 ? (
            <div style={{marginTop: 6, fontSize: 12}}>
              <span className="muted">People seen here: </span>
              {camSubs.map(function(s) {
                return (
                  <span key={s.id} className="chip" style={{background:'#1e1b4b', color:'#a5b4fc', marginRight:4, fontSize:11}}>
                    {s.name || s.id}
                  </span>
                );
              })}
            </div>
          ) : <div className="muted" style={{marginTop:4, fontSize:12}}>No re-ID subjects seen yet</div>}
        </div>
      )}
    </div>
  );
}
