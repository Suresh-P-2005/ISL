// ─────────────────────────────────────────────────────────────
// SPATIAL 3D, COLOR THEMES & LIGHT/DARK MODE MANAGER
// ─────────────────────────────────────────────────────────────
const ThemeManager = {
  init() {
    const savedTheme = localStorage.getItem('isl_favorite_theme') || 'aurora';
    const savedMode  = localStorage.getItem('isl_color_mode') || 'dark';

    this.setTheme(savedTheme);
    this.setMode(savedMode);

    // NOTE: Theme selector dropdown has been removed from all pages.
    // Theme is now set programmatically from localStorage on load only.

    const modeBtn = document.getElementById('theme-mode-btn');
    if (modeBtn) {
      modeBtn.addEventListener('click', () => this.toggleMode());
    }

    // Auth & Navigation Protection — inject user badge only (redirect handled by auth.js)
    this.updateAuthNavUI();

    // Navigation Menu Drawer Controller
    const menuBtn = document.getElementById('menu-btn');
    const closeBtn = document.getElementById('close-btn');
    const drawer = document.getElementById('drawer');
    const backdrop = document.getElementById('drawer-backdrop');

    if (menuBtn && drawer && backdrop) {
      menuBtn.addEventListener('click', () => {
        drawer.classList.add('active');
        backdrop.classList.add('active');
      });
    }
    if (closeBtn && drawer && backdrop) {
      closeBtn.addEventListener('click', () => {
        drawer.classList.remove('active');
        backdrop.classList.remove('active');
      });
    }
    if (backdrop && drawer) {
      backdrop.addEventListener('click', () => {
        drawer.classList.remove('active');
        backdrop.classList.remove('active');
      });
    }
  },

  updateAuthNavUI() {
    let session = null;
    try {
      session = JSON.parse(localStorage.getItem('isl_session') || 'null');
    } catch(e) {}

    // NOTE: Auth redirect is handled exclusively by auth.js requireAuth().
    // This function only injects the user badge UI — no redirects here.

    const isAdmin = session && session.role === 'ADMIN';

    // Hide /collect link if user is not ADMIN
    document.querySelectorAll('a[href="/collect"]').forEach(el => {
      el.style.display = isAdmin ? '' : 'none';
    });

    // Inject User Badge & Logout in navbar right section
    const navRight = document.querySelector('.nav-right');
    if (navRight && session && !document.getElementById('user-badge-wrap')) {
      const userWrap = document.createElement('div');
      userWrap.id = 'user-badge-wrap';
      userWrap.style.cssText = 'display:inline-flex;align-items:center;gap:8px;padding:4px 10px;border-radius:12px;background:var(--bg-glass);border:1px solid var(--border-mid);font-size:0.75rem;font-weight:700;color:var(--text-primary);';
      userWrap.innerHTML = `
        <span class="material-icons-round" style="font-size:16px;color:var(--brand-primary)">account_circle</span>
        <span>${session.username}</span>
        <span style="font-size:0.65rem;padding:2px 6px;border-radius:6px;background:${isAdmin?'rgba(239,68,68,0.2)':'rgba(59,130,246,0.2)'};color:${isAdmin?'#f87171':'#60a5fa'}">${session.role}</span>
        <button id="logout-btn" title="Log Out" style="background:none;border:none;color:var(--text-muted);cursor:pointer;display:flex;align-items:center;padding:2px;"><span class="material-icons-round" style="font-size:16px">logout</span></button>
      `;
      navRight.insertBefore(userWrap, navRight.firstChild);

      const logoutBtn = document.getElementById('logout-btn');
      if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
          localStorage.removeItem('isl_session');
          window.location.href = '/login';
        });
      }
    }
  },

  setTheme(themeId) {
    document.documentElement.setAttribute('data-theme', themeId);
    localStorage.setItem('isl_favorite_theme', themeId);

    // If spatial-light selected, sync mode to light
    if (themeId === 'spatial-light') {
      this.setMode('light');
    }
  },

  setMode(mode) {
    document.documentElement.setAttribute('data-mode', mode);
    localStorage.setItem('isl_color_mode', mode);

    const modeBtn = document.getElementById('theme-mode-btn');
    const modeIcon = document.getElementById('theme-mode-icon');

    if (modeIcon) {
      modeIcon.textContent = mode === 'dark' ? 'light_mode' : 'dark_mode';
    }
    if (modeBtn) {
      modeBtn.setAttribute('title', mode === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode');
      modeBtn.setAttribute('aria-label', mode === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode');
    }
  },

  toggleMode() {
    const current = document.documentElement.getAttribute('data-mode') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    this.setMode(next);
  }
};

// Ambient Mouse Light Tracking Spotlight
document.addEventListener('mousemove', (e) => {
  document.documentElement.style.setProperty('--mouse-x', `${e.clientX}px`);
  document.documentElement.style.setProperty('--mouse-y', `${e.clientY}px`);
});

document.addEventListener('DOMContentLoaded', () => {
  ThemeManager.init();
});
