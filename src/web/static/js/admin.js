document.addEventListener("DOMContentLoaded", () => {
  const session = JSON.parse(localStorage.getItem('isl_session'));
  if (!session || session.role !== 'ADMIN') {
    window.location.href = '/';
    return;
  }

  const token = session.token;
  const tbody = document.getElementById('user-table-body');
  const searchInput = document.getElementById('user-search');
  const refreshBtn = document.getElementById('refresh-btn');

  let allUsers = [];

  // Modals (we dynamically create them to keep HTML clean)
  const modalOverlay = document.createElement('div');
  modalOverlay.className = 'modal-overlay';
  modalOverlay.innerHTML = `
    <div class="modal-card">
      <div class="modal-title" id="modal-title">Confirm Action</div>
      <div class="modal-text" id="modal-text">Are you sure you want to proceed?</div>
      <div class="modal-actions">
        <button class="btn-cancel" id="modal-cancel">Cancel</button>
        <button class="btn-confirm" id="modal-confirm">Confirm</button>
      </div>
    </div>
  `;
  document.body.appendChild(modalOverlay);

  const btnCancel = document.getElementById('modal-cancel');
  const btnConfirm = document.getElementById('modal-confirm');
  
  let currentAction = null;

  btnCancel.addEventListener('click', () => {
    modalOverlay.classList.remove('active');
    currentAction = null;
  });

  btnConfirm.addEventListener('click', async () => {
    if (currentAction) {
      await currentAction();
    }
    modalOverlay.classList.remove('active');
  });

  function showModal(title, text, isDanger, onConfirm) {
    document.getElementById('modal-title').innerText = title;
    document.getElementById('modal-text').innerText = text;
    if (isDanger) {
      btnConfirm.classList.add('danger');
    } else {
      btnConfirm.classList.remove('danger');
    }
    currentAction = onConfirm;
    modalOverlay.classList.add('active');
  }

  async function fetchUsers() {
    try {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:2rem;">Loading users...</td></tr>';
      const res = await fetch('/api/v1/auth/users', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.status === 401 || res.status === 403) {
        localStorage.removeItem('isl_session');
        window.location.href = '/login';
        return;
      }
      const data = await res.json();
      if (data.status === 'ok') {
        allUsers = data.users;
        renderUsers(allUsers);
      }
    } catch (e) {
      console.error(e);
      tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--danger)">Failed to load users.</td></tr>';
    }
  }

  function renderUsers(users) {
    tbody.innerHTML = '';
    if (users.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No users found.</td></tr>';
      return;
    }

    users.forEach(u => {
      const tr = document.createElement('tr');
      const isRootAdmin = u.username.toLowerCase() === 'admin';
      const isCurrentUser = u.username === session.username;

      tr.innerHTML = `
        <td style="font-family:var(--font-mono);font-size:0.75rem;color:var(--text-muted)">#${u.id}</td>
        <td style="font-weight:600;">${u.username} ${isCurrentUser ? '<span style="color:var(--brand-primary);font-size:0.6rem;margin-left:4px;">(YOU)</span>' : ''}</td>
        <td>${u.email}</td>
        <td><span class="role-badge ${u.role === 'ADMIN' ? 'admin' : 'user'}">${u.role}</span></td>
        <td style="font-family:var(--font-mono);font-size:0.75rem;color:var(--text-muted)">${u.created_at.split(' ')[0]}</td>
        <td class="text-right">
          <div class="action-buttons">
            ${u.role === 'USER' ? 
              `<button class="btn-icon promote" title="Promote to Admin" data-id="${u.id}" data-name="${u.username}"><span class="material-icons-round">security</span></button>` : 
              `<button class="btn-icon" title="Revoke Admin" data-id="${u.id}" data-name="${u.username}" ${isRootAdmin ? 'disabled style="opacity:0.3"' : ''}><span class="material-icons-round">person</span></button>`
            }
            <button class="btn-icon delete" title="Delete User" data-id="${u.id}" data-name="${u.username}" ${isRootAdmin || isCurrentUser ? 'disabled style="opacity:0.3"' : ''}><span class="material-icons-round">delete</span></button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });

    // Bind events
    document.querySelectorAll('.btn-icon.promote').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const id = e.currentTarget.getAttribute('data-id');
        const name = e.currentTarget.getAttribute('data-name');
        showModal('Promote to Admin', `Are you sure you want to promote ${name} to Administrator?`, false, () => updateUserRole(id, 'ADMIN'));
      });
    });

    document.querySelectorAll('.btn-icon:not(.promote):not(.delete)').forEach(btn => {
      btn.addEventListener('click', (e) => {
        if(e.currentTarget.disabled) return;
        const id = e.currentTarget.getAttribute('data-id');
        const name = e.currentTarget.getAttribute('data-name');
        showModal('Revoke Admin', `Are you sure you want to revoke Administrator access from ${name}?`, true, () => updateUserRole(id, 'USER'));
      });
    });

    document.querySelectorAll('.btn-icon.delete').forEach(btn => {
      btn.addEventListener('click', (e) => {
        if(e.currentTarget.disabled) return;
        const id = e.currentTarget.getAttribute('data-id');
        const name = e.currentTarget.getAttribute('data-name');
        showModal('Delete Account', `WARNING: This action is permanent. Are you sure you want to delete ${name}'s account?`, true, () => deleteUser(id));
      });
    });
  }

  async function updateUserRole(userId, role) {
    try {
      const res = await fetch(`/api/v1/auth/users/${userId}/role`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ role })
      });
      if (res.ok) fetchUsers();
      else alert("Failed to update role.");
    } catch(e) { console.error(e); }
  }

  async function deleteUser(userId) {
    try {
      const res = await fetch(`/api/v1/auth/users/${userId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) fetchUsers();
      else alert("Failed to delete user.");
    } catch(e) { console.error(e); }
  }

  searchInput.addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase();
    const filtered = allUsers.filter(u => u.username.toLowerCase().includes(q) || u.email.toLowerCase().includes(q));
    renderUsers(filtered);
  });

  refreshBtn.addEventListener('click', fetchUsers);

  // Initial load
  fetchUsers();
});
