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

    // Inject User Badge & Logout in drawer nav
    const drawerNav = document.querySelector('.drawer-nav');
    if (drawerNav && session && !document.getElementById('drawer-user-badge')) {
      // Inject Admin Panel Link if user is ADMIN
      if (isAdmin && !document.getElementById('drawer-admin-link')) {
        const adminLink = document.createElement('a');
        adminLink.id = 'drawer-admin-link';
        adminLink.href = '/admin';
        adminLink.className = 'drawer-link';
        adminLink.innerHTML = `<span class="material-icons-round">security</span> Administration`;
        // Insert right after the last standard link, but before the divider
        drawerNav.appendChild(adminLink);
      }

      const divider = document.createElement('div');
      divider.style.cssText = 'height:1px;background:var(--border-subtle);margin:12px 0 4px 0;';
      
      const userWrap = document.createElement('div');
      userWrap.id = 'drawer-user-badge';
      userWrap.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-radius:var(--r-md);background:var(--bg-glass);border:1px solid var(--border-mid);margin-top:4px;';
      
      userWrap.innerHTML = `
        <div style="display:flex;align-items:center;gap:12px;">
          <span class="material-icons-round" style="font-size:24px;color:var(--brand-primary)">account_circle</span>
          <div style="display:flex;flex-direction:column;line-height:1.2;">
            <span style="font-size:0.9rem;font-weight:700;color:var(--text-primary);">${session.username}</span>
            <span style="font-size:0.65rem;font-weight:700;text-transform:uppercase;color:${isAdmin?'#f87171':'#60a5fa'}">${session.role}</span>
          </div>
        </div>
        <button id="logout-btn" title="Log Out" style="background:var(--danger-bg);border:1px solid var(--danger-border);color:var(--danger);cursor:pointer;display:flex;align-items:center;justify-content:center;padding:6px;border-radius:8px;transition:all 0.2s;">
          <span class="material-icons-round" style="font-size:18px">logout</span>
        </button>
      `;
      
      drawerNav.appendChild(divider);
      drawerNav.appendChild(userWrap);

      const logoutBtn = document.getElementById('logout-btn');
      if (logoutBtn) {
        logoutBtn.addEventListener('mouseenter', () => { logoutBtn.style.background = 'var(--danger)'; logoutBtn.style.color = '#fff'; });
        logoutBtn.addEventListener('mouseleave', () => { logoutBtn.style.background = 'var(--danger-bg)'; logoutBtn.style.color = 'var(--danger)'; });
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
