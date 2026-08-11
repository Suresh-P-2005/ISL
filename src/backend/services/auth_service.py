import os
import sqlite3
import hashlib
import hmac
import time
import json
import base64

PBKDF2_ITERATIONS = 100000

class AuthService:
    def __init__(self, db_dir: str, secret_key: str):
        self.secret_key = secret_key
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "users.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'USER',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_username ON users(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_email ON users(email)")

            # Auto-seed default Admin account if not exists
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = 'admin'")
            if not cursor.fetchone():
                salt, pwd_hash = self.hash_password("admin123")
                conn.execute(
                    "INSERT INTO users (username, email, password_hash, salt, role) VALUES (?, ?, ?, ?, ?)",
                    ("admin", "admin@isl.local", pwd_hash, salt, "ADMIN")
                )

    def hash_password(self, password: str, salt: str = None) -> tuple:
        if not salt:
            salt = os.urandom(16).hex()
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            PBKDF2_ITERATIONS
        ).hex()
        return salt, pwd_hash

    def register_user(self, username: str, email: str, password: str, role: str = "USER") -> dict:
        username = username.strip()
        email = email.strip().lower()
        if not username or not email or not password:
            raise ValueError("Username, email, and password are required.")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters.")

        salt, pwd_hash = self.hash_password(password)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, salt, role) VALUES (?, ?, ?, ?, ?)",
                    (username, email, pwd_hash, salt, role)
                )
                user_id = cursor.lastrowid
                return {
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "role": role
                }
        except sqlite3.IntegrityError:
            raise ValueError("Username or Email already exists.")

    def authenticate_user(self, username_or_email: str, password: str) -> dict:
        target = username_or_email.strip().lower()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?",
                (target, target)
            )
            user = cursor.fetchone()
            if not user:
                raise ValueError("Invalid username/email or password.")

            _, calc_hash = self.hash_password(password, user["salt"])
            if not hmac.compare_digest(calc_hash, user["password_hash"]):
                raise ValueError("Invalid username/email or password.")

            token = self.create_jwt_token(user["id"], user["username"], user["role"])
            return {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"],
                "token": token
            }

    def create_jwt_token(self, user_id: int, username: str, role: str) -> str:
        header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
        payload_data = {
            "sub": user_id,
            "username": username,
            "role": role,
            "exp": int(time.time()) + (86400 * 7) # 7 days
        }
        payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).decode().rstrip("=")
        signature_input = f"{header}.{payload}"
        signature = hmac.new(self.secret_key.encode(), signature_input.encode(), hashlib.sha256).digest()
        sig_str = base64.urlsafe_b64encode(signature).decode().rstrip("=")
        return f"{header}.{payload}.{sig_str}"

    def verify_jwt_token(self, token: str) -> dict:
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            header, payload, sig_str = parts
            signature_input = f"{header}.{payload}"
            calc_sig = base64.urlsafe_b64encode(
                hmac.new(self.secret_key.encode(), signature_input.encode(), hashlib.sha256).digest()
            ).decode().rstrip("=")
            if not hmac.compare_digest(calc_sig, sig_str):
                return None

            rem = len(payload) % 4
            if rem > 0:
                payload += "=" * (4 - rem)
            data = json.loads(base64.urlsafe_b64decode(payload).decode())
            if data.get("exp", 0) < time.time():
                return None
            return data
        except Exception:
            return None

    def get_all_users(self) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email, role, created_at FROM users ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def update_user_role(self, user_id: int, new_role: str) -> bool:
        if new_role not in ["ADMIN", "USER"]:
            raise ValueError("Invalid role specified.")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Prevent removing the only admin if possible, but for now just update.
            cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
            if cursor.rowcount == 0:
                raise ValueError("User not found.")
            return True

    def delete_user(self, user_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Prevent deleting the default 'admin' account to ensure the system is always accessible
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row and row[0].lower() == 'admin':
                raise ValueError("Cannot delete the root admin account.")
            
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            if cursor.rowcount == 0:
                raise ValueError("User not found.")
            return True
