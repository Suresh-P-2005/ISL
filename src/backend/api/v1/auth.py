import logging
from fastapi import APIRouter, Request, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

auth_router = APIRouter(tags=["Authentication"])

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class RoleUpdateRequest(BaseModel):
    role: str

@auth_router.post("/api/v1/auth/register")
async def register(payload: RegisterRequest, request: Request):
    auth_service = request.app.state.auth_service
    logger.info(f"Register attempt: username={payload.username}, email={payload.email}")
    try:
        user = auth_service.register_user(
            username=payload.username,
            email=payload.email,
            password=payload.password,
            role="USER"
        )
        logger.info(f"Register SUCCESS: {payload.username}")
        return {"status": "ok", "message": "User registered successfully. Please login.", "user": user}
    except ValueError as e:
        logger.warning(f"Register FAIL: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@auth_router.post("/api/v1/auth/login")
async def login(payload: LoginRequest, request: Request):
    auth_service = request.app.state.auth_service
    logger.info(f"Login attempt: username={payload.username}")
    try:
        res = auth_service.authenticate_user(
            username_or_email=payload.username,
            password=payload.password
        )
        logger.info(f"Login SUCCESS: {payload.username}")
        return {"status": "ok", "user": res}
    except ValueError as e:
        logger.warning(f"Login FAIL: {e}")
        raise HTTPException(status_code=401, detail=str(e))

@auth_router.get("/api/v1/auth/me")
async def get_me(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]
    auth_service = request.app.state.auth_service
    payload = auth_service.verify_jwt_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")
    return {"status": "ok", "user": payload}

def verify_admin(request: Request, authorization: str):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]
    payload = request.app.state.auth_service.verify_jwt_token(token)
    if not payload or payload.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Forbidden. Admin access required.")
    return payload

@auth_router.get("/api/v1/auth/users")
async def get_users(request: Request, authorization: str = Header(None)):
    verify_admin(request, authorization)
    users = request.app.state.auth_service.get_all_users()
    return {"status": "ok", "users": users}

@auth_router.put("/api/v1/auth/users/{user_id}/role")
async def update_user_role(user_id: int, payload: RoleUpdateRequest, request: Request, authorization: str = Header(None)):
    verify_admin(request, authorization)
    try:
        request.app.state.auth_service.update_user_role(user_id, payload.role.upper())
        return {"status": "ok", "message": "User role updated successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@auth_router.delete("/api/v1/auth/users/{user_id}")
async def delete_user(user_id: int, request: Request, authorization: str = Header(None)):
    verify_admin(request, authorization)
    try:
        request.app.state.auth_service.delete_user(user_id)
        return {"status": "ok", "message": "User deleted successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
