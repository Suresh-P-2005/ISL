/* ═══════════════════════════════════════════════════════════
   ISL TRANSLATOR — AUTHENTICATION & SESSION MANAGER
═══════════════════════════════════════════════════════════ */

'use strict';

class AuthManager {
  static getSession() {
    try {
      const s = localStorage.getItem('isl_session');
      return s ? JSON.parse(s) : null;
    } catch(e) { return null; }
  }

  static setSession(userData) {
    try {
      localStorage.setItem('isl_session', JSON.stringify(userData));
    } catch(e) {}
  }

  static logout() {
    localStorage.removeItem('isl_session');
    window.location.href = '/login';
  }

  static async verifySessionWithBackend() {
    const session = this.getSession();
    if (!session || !session.token) return false;
    
    try {
      const BACKEND = window.location.origin;
      const res = await fetch(BACKEND + '/api/v1/auth/me', {
        headers: { 'Authorization': 'Bearer ' + session.token }
      });
      return res.ok;
    } catch (e) {
      return false;
    }
  }

  static async requireAuth() {
    const session = AuthManager.getSession();
    const currentPath = window.location.pathname;

    // 1. Initial frontend check
    if (!session || !session.token) {
      if (currentPath !== '/login') {
        window.location.href = '/login?redirect=' + encodeURIComponent(currentPath);
      }
      return;
    }

    // 2. Deep backend verification
    const isValid = await this.verifySessionWithBackend();
    
    if (!isValid) {
      this.logout();
      return;
    }

    // 3. Valid session handling
    if (currentPath === '/login') {
      window.location.href = '/';
      return;
    }

    // Restrict /collect to ADMIN only
    if (currentPath === '/collect' && session.role !== 'ADMIN') {
      document.body.innerHTML = `
        <div style="min-height:100vh;display:flex;align-items:center;justify-content:center;background:#090d16;color:#fff;font-family:sans-serif;padding:2rem;text-align:center;">
          <div style="max-width:480px;background:rgba(255,255,255,0.05);padding:2rem;border-radius:16px;border:1px solid rgba(239,68,68,0.3);">
            <div style="font-size:3rem;margin-bottom:1rem;">⛔</div>
            <h1 style="color:#ef4444;font-size:1.5rem;margin-bottom:0.5rem;">Access Denied</h1>
            <p style="color:#94a3b8;font-size:0.9rem;margin-bottom:1.5rem;">Administrator rights are required to access the Data Collector Studio.</p>
            <a href="/" style="display:inline-block;padding:10px 20px;background:#3b82f6;color:#fff;text-decoration:none;border-radius:8px;font-weight:600;">Return to Real-Time Vision</a>
          </div>
        </div>`;
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const BACKEND = window.location.origin;
  AuthManager.requireAuth();

  const card3d = document.getElementById('auth-3d-card');
  const loginForm = document.getElementById('login-form');
  const registerForm = document.getElementById('register-form');
  const alertFront = document.getElementById('alert-banner-front');
  const alertBack = document.getElementById('alert-banner-back');

  // Enforce zero auto-filling on startup
  clearInputs();

  // Password Visibility Toggle Handlers
  setupPassToggle('toggle-login-pass', 'login-password');
  setupPassToggle('toggle-reg-pass', 'reg-password');

  function setupPassToggle(btnId, inputId) {
    const btn = document.getElementById(btnId);
    const input = document.getElementById(inputId);
    if (btn && input) {
      btn.addEventListener('click', () => {
        const isPass = input.type === 'password';
        input.type = isPass ? 'text' : 'password';
        const icon = btn.querySelector('.material-icons-round');
        if (icon) icon.textContent = isPass ? 'visibility_off' : 'visibility';
      });
    }
  }

  // 3D Card flip — "Create an Account" link
  const btnGotoRegister = document.getElementById('btn-goto-register');
  const btnGotoLogin    = document.getElementById('btn-goto-login');

  if (card3d) {
    if (btnGotoRegister) {
      btnGotoRegister.addEventListener('click', (e) => {
        e.preventDefault();
        card3d.classList.add('flipped');
        showAlert('', 'err', alertFront);
        showAlert('', 'err', alertBack);
      });
    }
    if (btnGotoLogin) {
      btnGotoLogin.addEventListener('click', (e) => {
        e.preventDefault();
        card3d.classList.remove('flipped');
        showAlert('', 'err', alertFront);
        showAlert('', 'err', alertBack);
      });
    }
  }

  // ── LOGIN FORM ──────────────────────────────────────────────
  if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const username = document.getElementById('login-username').value.trim();
      const password = document.getElementById('login-password').value;
      const loginBtn = document.getElementById('login-btn');

      setLoading(loginBtn, true);
      showAlert('', 'err', alertFront);

      try {
        const r = await fetch(BACKEND + '/api/v1/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password })
        });
        const data = await r.json();
        if (r.ok && data.status === 'ok') {
          AuthManager.setSession(data.user);
          showAlert('✓ Login successful! Redirecting…', 'ok', alertFront);
          setTimeout(() => {
            const params = new URLSearchParams(window.location.search);
            window.location.href = params.get('redirect') || '/';
          }, 800);
        } else {
          showAlert(data.detail || 'Login failed. Check your credentials.', 'err', alertFront);
          setLoading(loginBtn, false);
        }
      } catch(err) {
        showAlert('Server connection failed. Please try again.', 'err', alertFront);
        setLoading(loginBtn, false);
      }
    });
  }

  // ── REGISTER FORM ───────────────────────────────────────────
  if (registerForm) {
    registerForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const username = document.getElementById('reg-username').value.trim();
      const email    = document.getElementById('reg-email').value.trim();
      const password = document.getElementById('reg-password').value;
      const regBtn   = document.getElementById('reg-btn');

      setLoading(regBtn, true);
      showAlert('', 'err', alertBack);

      try {
        const r = await fetch(BACKEND + '/api/v1/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, email, password })
        });
        const data = await r.json();
        if (r.ok && data.status === 'ok') {
          // Show success on back face first, then flip to login after delay
          showAlert('✓ Account created! Please sign in.', 'ok', alertBack);
          clearInputs();
          setTimeout(() => {
            if (card3d) card3d.classList.remove('flipped');
            showAlert('✓ Account created! Please sign in now.', 'ok', alertFront);
          }, 1400);
        } else {
          showAlert(data.detail || 'Registration failed.', 'err', alertBack);
          setLoading(regBtn, false);
        }
      } catch(err) {
        showAlert('Server connection failed. Please try again.', 'err', alertBack);
        setLoading(regBtn, false);
      }
    });
  }

  // ── HELPERS ─────────────────────────────────────────────────

  function clearInputs() {
    ['login-username', 'login-password', 'reg-username', 'reg-email', 'reg-password'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.value = '';
    });
  }

  function showAlert(msg, type = 'err', targetBanner = alertFront) {
    const banner = targetBanner || alertFront;
    if (!banner) return;
    if (!msg) {
      banner.style.display = 'none';
      return;
    }
    banner.className = `alert-banner ${type}`;
    banner.textContent = msg;
    banner.style.display = 'block';
  }

  function setLoading(btn, isLoading) {
    if (!btn) return;
    btn.disabled = isLoading;
    btn.style.opacity = isLoading ? '0.7' : '1';
    btn.style.cursor  = isLoading ? 'wait' : 'pointer';
  }
});
