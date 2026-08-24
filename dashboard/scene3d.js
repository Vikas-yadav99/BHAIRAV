// Phase 21: 3D Scene Visualization (Three.js)
// Loads camera positions, persons, zones from /api/scene3d and renders
// a real-time 3D scene with camera frustums, person markers, and zone overlays.

function initScene3D(container, token) {
  // Dynamic import of Three.js from CDN
  const script = document.createElement('script');
  script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
  script.onload = () => buildScene(container, token);
  document.head.appendChild(script);
}

function buildScene(container, token) {
  const THREE = window.THREE;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e17);
  const camera3d = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera3d.position.set(0, 30, 40);
  camera3d.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  // Ground plane
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(100, 100),
    new THREE.MeshStandardMaterial({ color: 0x111827, transparent: true, opacity: 0.8 })
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  // Grid
  const grid = new THREE.GridHelper(100, 50, 0x21262d, 0x161b22);
  scene.add(grid);

  // Lights
  scene.add(new THREE.AmbientLight(0x404060, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(20, 30, 10);
  scene.add(dir);

  // Person markers (spheres)
  const personMeshes = {};
  const personGeo = new THREE.SphereGeometry(0.3, 8, 8);
  const alertMat = new THREE.MeshStandardMaterial({ color: 0xf85149, emissive: 0xf85149, emissiveIntensity: 0.5 });
  const normalMat = new THREE.MeshStandardMaterial({ color: 0x3fb950, emissive: 0x3fb950, emissiveIntensity: 0.3 });

  // Camera frustum cones
  const camMeshes = {};

  // Zone polygons (flat shapes on ground)
  const zoneMeshes = [];

  // Controls (simple orbit)
  let isDragging = false, prevMouse = { x: 0, y: 0 };
  let camAngle = 0, camHeight = 30, camDist = 40;

  container.addEventListener('mousedown', e => { isDragging = true; prevMouse = { x: e.clientX, y: e.clientY }; });
  container.addEventListener('mouseup', () => isDragging = false);
  container.addEventListener('mouseleave', () => isDragging = false);
  container.addEventListener('mousemove', e => {
    if (!isDragging) return;
    camAngle -= (e.clientX - prevMouse.x) * 0.005;
    camHeight = Math.max(5, Math.min(80, camHeight + (e.clientY - prevMouse.y) * 0.2));
    prevMouse = { x: e.clientX, y: e.clientY };
  });
  container.addEventListener('wheel', e => {
    camDist = Math.max(10, Math.min(100, camDist + e.deltaY * 0.05));
  });

  function animate() {
    requestAnimationFrame(animate);
    camera3d.position.x = Math.sin(camAngle) * camDist;
    camera3d.position.z = Math.cos(camAngle) * camDist;
    camera3d.position.y = camHeight;
    camera3d.lookAt(0, 0, 0);
    renderer.render(scene, camera3d);
  }
  animate();

  // Fetch and update scene data
  async function updateScene() {
    try {
      const resp = await fetch('/api/scene3d', { headers: { 'Authorization': 'Bearer ' + token } });
      if (!resp.ok) return;
      const data = await resp.json();

      // Update persons
      const seen = new Set();
      for (const p of data.persons || []) {
        seen.add(p.track_id);
        if (!personMeshes[p.track_id]) {
          const mat = p.alert ? alertMat : normalMat;
          personMeshes[p.track_id] = new THREE.Mesh(personGeo, mat);
          scene.add(personMeshes[p.track_id]);
        }
        personMeshes[p.track_id].position.set(p.position[0] - 50, 0.5, p.position[2] - 50);
        personMeshes[p.track_id].material = p.alert ? alertMat : normalMat;
      }
      for (const id of Object.keys(personMeshes)) {
        if (!seen.has(Number(id))) {
          scene.remove(personMeshes[id]);
          delete personMeshes[id];
        }
      }

      // Update cameras (add cones once)
      for (const c of data.cameras || []) {
        if (!camMeshes[c.id]) {
          const coneGeo = new THREE.ConeGeometry(3, 8, 4, 1, true);
          const coneMat = new THREE.MeshStandardMaterial({ color: 0x58a6ff, transparent: true, opacity: 0.15, side: THREE.DoubleSide });
          const cone = new THREE.Mesh(coneGeo, coneMat);
          cone.position.set(c.position[0] - 50, c.position[1], c.position[2] - 50);
          cone.rotation.x = THREE.MathUtils.degToRad(c.rotation[0]);
          cone.rotation.y = THREE.MathUtils.degToRad(c.rotation[1]);
          scene.add(cone);
          camMeshes[c.id] = cone;
        }
      }

      // Update zones
      for (const z of data.zones || []) {
        const shape = new THREE.Shape();
        if (z.points && z.points.length > 0) {
          shape.moveTo(z.points[0][0] * 100 - 50, z.points[0][1] * 100 - 50);
          for (let i = 1; i < z.points.length; i++) {
            shape.lineTo(z.points[i][0] * 100 - 50, z.points[i][1] * 100 - 50);
          }
          shape.closePath();
          const geo = new THREE.ShapeGeometry(shape);
          const mat = new THREE.MeshBasicMaterial({ color: parseInt(z.color.slice(1, 7), 16), transparent: true, opacity: 0.2, side: THREE.DoubleSide });
          const mesh = new THREE.Mesh(geo, mat);
          mesh.rotation.x = -Math.PI / 2;
          mesh.position.y = 0.05;
          scene.add(mesh);
          zoneMeshes.push(mesh);
        }
      }
    } catch (_) {}
  }

  updateScene();
  setInterval(updateScene, 1000);
  return { scene, camera: camera3d, renderer };
}
