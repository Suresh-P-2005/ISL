/* ═══════════════════════════════════════════════════════════
   ISL TRANSLATOR — 3D LIVE SHOWCASE CONTROLLER
═══════════════════════════════════════════════════════════ */

'use strict';

document.addEventListener('DOMContentLoaded', () => {
  const menuBtn = document.getElementById('menu-btn');
  const closeBtn = document.getElementById('close-btn');
  const drawer = document.getElementById('drawer');
  const backdrop = document.getElementById('drawer-backdrop');

  if (menuBtn && drawer && backdrop) {
    menuBtn.addEventListener('click', () => {
      drawer.classList.add('active'); backdrop.classList.add('active');
    });
  }
  if (closeBtn && drawer && backdrop) {
    closeBtn.addEventListener('click', () => {
      drawer.classList.remove('active'); backdrop.classList.remove('active');
    });
  }
  if (backdrop && drawer) {
    backdrop.addEventListener('click', () => {
      drawer.classList.remove('active'); backdrop.classList.remove('active');
    });
  }

  Polyhedron3D.init('bg-canvas');
});

const Polyhedron3D = {
  init(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    const nodes = [
      [-100, -100, -100], [100, -100, -100], [100, 100, -100], [-100, 100, -100],
      [-100, -100, 100], [100, -100, 100], [100, 100, 100], [-100, 100, 100]
    ];
    const edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]];
    let angle = 0;

    function animate() {
      ctx.clearRect(0, 0, width, height);
      angle += 0.008;

      const proj = [];
      const cosA = Math.cos(angle), sinA = Math.sin(angle);
      const cx = width / 2, cy = height / 2;

      for (let i = 0; i < nodes.length; i++) {
        let [nx, ny, nz] = nodes[i];
        let rx = nx * cosA - nz * sinA;
        let rz = nx * sinA + nz * cosA;
        let ry = ny * cosA - rz * sinA;
        let rzFinal = ny * sinA + rz * cosA;

        let fov = 400 / (400 + rzFinal);
        let px = cx + rx * fov;
        let py = cy + ry * fov;
        proj.push([px, py]);

        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--brand-secondary').trim() || '#34d399';
        ctx.fill();
      }

      for (let i = 0; i < edges.length; i++) {
        const [e1, e2] = edges[i];
        const [x1, y1] = proj[e1];
        const [x2, y2] = proj[e2];
        ctx.beginPath();
        ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--brand-primary').trim() || '#38bdf8';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      requestAnimationFrame(animate);
    }

    window.addEventListener('resize', () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    });

    animate();
  }
};
