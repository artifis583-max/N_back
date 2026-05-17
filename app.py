#neo

import os
import hmac
import re
import threading
import time
import asyncio
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from fastapi import FastAPI, HTTPException, Request, Response
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse, PlainTextResponse
# from fastapi.templating import Jinja2Templates

# templates = Jinja2Templates(directory="templates")

BASE_DIR = Path(__file__).resolve().parent


# ── Config ────────────────────────────────────────────────────────────────────
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
CLIENT_HEADER = os.getenv("AUTH_CLIENT_HEADER", "x-client-key").lower()
ADMIN_HEADER = os.getenv("AUTH_ADMIN_HEADER", "x-admin-key").lower()
CLIENT_SHARED_KEY = os.getenv("AUTH_CLIENT_SHARED_KEY", "")
ADMIN_SHARED_KEY = os.getenv("AUTH_ADMIN_SHARED_KEY", "")
JWT_SECRET = os.getenv("AUTH_JWT_SECRET", "")
# Separate secret for admin sessions — if unset, falls back to JWT_SECRET
ADMIN_JWT_SECRET = os.getenv("AUTH_ADMIN_JWT_SECRET", "") or JWT_SECRET
JWT_TTL_SECONDS = int(os.getenv("AUTH_JWT_TTL_SECONDS", "300"))
ADMIN_SESSION_TTL_SECONDS = int(os.getenv("AUTH_ADMIN_SESSION_TTL_SECONDS", "3600"))
ADMIN_COOKIE_SECURE = os.getenv("AUTH_ADMIN_COOKIE_SECURE", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "240"))
GEMINI_MAX_CONCURRENCY = int(os.getenv("GEMINI_MAX_CONCURRENCY", "50"))
GEMINI_QUEUE_TIMEOUT_SECONDS = float(os.getenv("GEMINI_QUEUE_TIMEOUT_SECONDS", "30"))

RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("AUTH_RATE_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("AUTH_RATE_MAX_REQUESTS", "60"))

# Admin login brute-force guard: 10 attempts per minute per IP
ADMIN_LOGIN_RATE_WINDOW = 60
ADMIN_LOGIN_RATE_MAX = 10

# ── Singletons ────────────────────────────────────────────────────────────────
_gemini_model: genai.GenerativeModel | None = None
_gemini_lock = threading.Lock()
_gemini_slots = threading.BoundedSemaphore(GEMINI_MAX_CONCURRENCY)
_bg_queue: asyncio.Queue | None = None
_db_pool: ConnectionPool | None = None


def _get_gemini(system_prompt: str) -> genai.GenerativeModel:
    """Return a Gemini GenerativeModel configured with the given system prompt."""
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
    )


# ── Background writer ─────────────────────────────────────────────────────────
async def _bg_writer(queue: asyncio.Queue) -> None:
    """Drain the write queue and commit DB inserts off the hot path."""
    while True:
        task = await queue.get()
        kind = task[0]
        try:
            if kind == "event":
                _, event, device_id, remaining, detail = task
                _do_append_event(event, device_id, remaining, detail)
            elif kind == "qa":
                (
                    _,
                    device_id,
                    question,
                    answer,
                    option_text,
                    raw_response,
                    input_mode,
                ) = task
                _do_log_qa(
                    device_id, question, answer, option_text, raw_response, input_mode
                )
        except Exception as exc:
            import sys

            print(f"[bg_writer] error processing {kind!r} task: {exc}", file=sys.stderr)
        finally:
            queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bg_queue, _db_pool
    _require_configured()
    _db_pool = ConnectionPool(
        DATABASE_URL, min_size=2, max_size=10, kwargs={"row_factory": dict_row}
    )
    _migrate()
    _bg_queue = asyncio.Queue(maxsize=2000)
    asyncio.create_task(_bg_writer(_bg_queue))
    yield
    _db_pool.close()


app = FastAPI(title="MCQ Solver API", lifespan=lifespan)


# ── Models ────────────────────────────────────────────────────────────────────
class AuthRequest(BaseModel):
    device_id: str = Field(min_length=12, max_length=128)
    app_version: str = Field(default="", max_length=32)  # sent by client on every auth
    # Optional PC info — sent only when client expects denial (not-yet-approved device)
    pc_name: str = Field(default="", max_length=256)
    pc_user: str = Field(default="", max_length=256)
    os_info: str = Field(default="", max_length=512)
    extra: str = Field(default="", max_length=1000)


class DeviceUpsert(BaseModel):
    device_id: str = Field(min_length=12, max_length=128)
    allowed: bool = True
    remaining: int = Field(ge=0)
    note: str = ""
    mode: str = "text"


class DevicePatch(BaseModel):
    allowed: bool | None = None
    remaining: int | None = Field(default=None, ge=0)
    note: str | None = None
    mode: str | None = None


class SolveRequest(BaseModel):
    text: str | None = Field(default=None, max_length=20000)
    image: str | None = Field(default=None)  # base64 PNG/JPEG (raw or data URI)


# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPT_TEXT = """
You are a competitive programming assistant.

The user will send extracted text from a coding problem interface.
Your job: read the problem, solve it, and return ONLY the raw solution code.

Rules:
- Output ONLY the complete, compilable code. No explanation, no markdown, no code fences.
- detect language and if header and footer present in Question then provide only required part of code otherwise full code:
- Do NOT indent any line. No leading spaces or tabs. Write all code flush to the left margin.
- If the problem specifies another language, use that language instead.
- Every line of code must be on its own line — do NOT collapse to one line.
- Do NOT output ANSWER:, OPTION:, or any label. Just the raw code.
- If no clear coding problem is found, output exactly one line:
  // No problem detected
"""

PROMPT_IMAGE = """
You are a competitive programming assistant.

The user will send a screenshot of a coding problem.
Your job: read the problem from the image, solve it, and return ONLY the raw solution code.

Rules:
- Output ONLY the complete, compilable code. No explanation, no markdown, no code fences.
- detect language and if header and footer present in Question then provide only required part of code otherwise full code:
- If the problem specifies another language, use that language instead.
- Do NOT indent any line. No leading spaces or tabs. Write all code flush to the left margin.
- Every line of code must be on its own line — do NOT collapse to one line.
- Do NOT output ANSWER:, OPTION:, or any label. Just the raw code.
- Ignore all UI chrome: timers, sidebars, navigation, leaderboards.
- If no clear coding problem is found, output exactly one line:
  // No problem detected
"""


# ── Database ──────────────────────────────────────────────────────────────────
@contextmanager
def _connect():
    with _db_pool.connection() as conn:
        yield conn


def _exec(conn, sql: str, params: tuple = ()):
    return conn.execute(sql, params)


def _now() -> int:
    return int(time.time())


def _row_to_dict(row) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _require_configured() -> None:
    missing = []
    if not ADMIN_PASSWORD:
        missing.append("ADMIN_PASSWORD")
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if not CLIENT_SHARED_KEY:
        missing.append("AUTH_CLIENT_SHARED_KEY")
    if not ADMIN_SHARED_KEY:
        missing.append("AUTH_ADMIN_SHARED_KEY")
    if not JWT_SECRET:
        missing.append("AUTH_JWT_SECRET")
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


def _migrate() -> None:
    with _connect() as conn:
        _exec(
            conn,
            """
            CREATE TABLE IF NOT EXISTS devices (
                device_id    TEXT    PRIMARY KEY,
                allowed      BOOLEAN NOT NULL DEFAULT TRUE,
                remaining    INTEGER NOT NULL DEFAULT 0 CHECK (remaining >= 0),
                note         TEXT    NOT NULL DEFAULT '',
                created_at   BIGINT  NOT NULL,
                updated_at   BIGINT  NOT NULL,
                last_seen_at BIGINT
            )
        """,
        )
        _exec(
            conn,
            """
            CREATE TABLE IF NOT EXISTS auth_events (
                id        BIGSERIAL PRIMARY KEY,
                ts        BIGINT NOT NULL,
                event     TEXT   NOT NULL,
                device_id TEXT,
                remaining INTEGER,
                detail    TEXT
            )
        """,
        )
        _exec(
            conn,
            """
            CREATE TABLE IF NOT EXISTS rate_limits (
                rate_key     TEXT   PRIMARY KEY,
                window_start BIGINT NOT NULL,
                count        INTEGER NOT NULL
            )
        """,
        )
        _exec(
            conn,
            """
            CREATE TABLE IF NOT EXISTS qa_logs (
                id          BIGSERIAL PRIMARY KEY,
                ts          BIGINT NOT NULL,
                device_id   TEXT   NOT NULL,
                input_mode  TEXT   NOT NULL DEFAULT 'text',
                question    TEXT   NOT NULL,
                answer      TEXT   NOT NULL,
                option_text TEXT   NOT NULL,
                raw_response TEXT  NOT NULL
            )
        """,
        )
        _exec(
            conn,
            "CREATE INDEX IF NOT EXISTS idx_auth_events_device_ts ON auth_events(device_id, ts)",
        )
        _exec(
            conn,
            "CREATE INDEX IF NOT EXISTS idx_qa_logs_device_ts     ON qa_logs(device_id, ts)",
        )
        _exec(
            conn,
            "ALTER TABLE qa_logs ADD COLUMN IF NOT EXISTS input_mode TEXT NOT NULL DEFAULT 'text'",
        )
        # Add mode column so per-device text/image preference is persisted
        _exec(
            conn,
            "ALTER TABLE devices ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'text'",
        )
        # Global key-value settings table (stores min_app_version, etc.)
        _exec(
            conn,
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL DEFAULT '',
                updated_at BIGINT NOT NULL
            )
            """,
        )
        # Seed a default minimum version — admin can update via /admin/settings
        _exec(
            conn,
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES ('min_app_version', '1.0.0', %s)
            ON CONFLICT (key) DO NOTHING
            """,
            (_now(),),
        )
        # Seed require_version_check — when 'true', clients with NO version are also blocked
        _exec(
            conn,
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES ('require_version_check', 'false', %s)
            ON CONFLICT (key) DO NOTHING
            """,
            (_now(),),
        )
        # Seed disclaimer text — shown to users on startup
        _exec(
            conn,
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES ('disclaimer', '⚠ Users are solely responsible for their actions and use of this application. The developers shall not be held liable for any misuse, damages, legal issues, or consequences arising from the use of this software.', %s)
            ON CONFLICT (key) DO NOTHING
            """,
            (_now(),),
        )
        # Table for denied/unauthorized device info (PC details collected for verification)
        _exec(
            conn,
            """
            CREATE TABLE IF NOT EXISTS denied_devices (
                id          BIGSERIAL PRIMARY KEY,
                ts          BIGINT NOT NULL,
                device_id   TEXT   NOT NULL,
                denial_type TEXT   NOT NULL,
                pc_name     TEXT   NOT NULL DEFAULT '',
                pc_user     TEXT   NOT NULL DEFAULT '',
                os_info     TEXT   NOT NULL DEFAULT '',
                app_version TEXT   NOT NULL DEFAULT '',
                extra       TEXT   NOT NULL DEFAULT ''
            )
            """,
        )
        _exec(
            conn,
            "CREATE INDEX IF NOT EXISTS idx_denied_devices_device ON denied_devices(device_id)",
        )


# ── App settings helpers ───────────────────────────────────────────────────────
def _get_setting(key: str, default: str = "") -> str:
    try:
        with _connect() as conn:
            row = _exec(
                conn, "SELECT value FROM app_settings WHERE key = %s", (key,)
            ).fetchone()
        return row["value"] if row else default
    except Exception:
        return default


def _log_denied_device(
    device_id: str,
    denial_type: str,
    pc_name: str = "",
    pc_user: str = "",
    os_info: str = "",
    app_version: str = "",
    extra: str = "",
) -> None:
    """Persist PC info for unauthorized/version-denied clients (for verification)."""
    try:
        with _connect() as conn:
            _exec(
                conn,
                """
                INSERT INTO denied_devices
                    (ts, device_id, denial_type, pc_name, pc_user, os_info, app_version, extra)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    _now(),
                    (device_id or "")[:128],
                    (denial_type or "")[:64],
                    (pc_name or "")[:256],
                    (pc_user or "")[:256],
                    (os_info or "")[:512],
                    (app_version or "")[:32],
                    (extra or "")[:1000],
                ),
            )
    except Exception:
        pass


def _set_setting(key: str, value: str) -> None:
    with _connect() as conn:
        _exec(
            conn,
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
            """,
            (key, value, _now()),
        )


def _parse_version(v: str) -> tuple[int, ...]:
    """Convert '1.2.3' → (1, 2, 3). Non-numeric parts become 0."""
    parts = []
    for p in str(v).strip().split(".")[:4]:
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts) or (0,)


def _version_expired(client_version: str) -> tuple[bool, str, str]:
    """
    Check whether a client should be blocked due to version policy.

    Returns (expired: bool, client_ver: str, min_ver: str).

    Two blocking scenarios:
      1. Client sent a version that is below min_app_version.
      2. Client sent NO version AND require_version_check == 'true'
         (covers old builds that pre-date the version field).
    """
    min_ver = _get_setting("min_app_version", "1.0.0")
    require_check = _get_setting("require_version_check", "false").lower() == "true"

    if not client_version:
        # Old client — no version sent at all
        if require_check:
            # Treat as version "0.0.0" so the popup shows a meaningful comparison
            return True, "0.0.0", min_ver
        return False, "", min_ver

    # Versioned client — compare normally
    expired = _parse_version(client_version) < _parse_version(min_ver)
    return expired, client_version, min_ver


# ── Device helpers ────────────────────────────────────────────────────────────
def get_device(device_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = _exec(
            conn, "SELECT * FROM devices WHERE device_id = %s", (device_id,)
        ).fetchone()
    device = _row_to_dict(row)
    if device is not None:
        device["allowed"] = bool(device["allowed"])
    return device


def list_devices() -> dict[str, dict[str, Any]]:
    with _connect() as conn:
        rows = _exec(conn, "SELECT * FROM devices ORDER BY updated_at DESC").fetchall()
    devices = {}
    for row in rows:
        device = _row_to_dict(row)
        device["allowed"] = bool(device["allowed"])
        devices[device["device_id"]] = device
    return devices


def upsert_device_record(
    device_id: str, allowed: bool, remaining: int, note: str = "", mode: str = "text"
) -> dict[str, Any]:
    now = _now()
    mode = mode if mode in ("text", "image") else "text"
    with _connect() as conn:
        # Postgres UPSERT — preserves created_at on conflict
        _exec(
            conn,
            """
            INSERT INTO devices (device_id, allowed, remaining, note, mode, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (device_id) DO UPDATE
                SET allowed    = EXCLUDED.allowed,
                    remaining  = EXCLUDED.remaining,
                    note       = EXCLUDED.note,
                    mode       = EXCLUDED.mode,
                    updated_at = EXCLUDED.updated_at
        """,
            (device_id, allowed, remaining, note, mode, now, now),
        )
    device = get_device(device_id)
    if device is None:
        raise RuntimeError("Device upsert failed")
    return device


def patch_device_record(
    device_id: str,
    *,
    allowed: bool | None = None,
    remaining: int | None = None,
    note: str | None = None,
    mode: str | None = None,
) -> dict[str, Any] | None:
    device = get_device(device_id)
    if device is None:
        return None
    allowed = device["allowed"] if allowed is None else allowed
    remaining = int(device["remaining"]) if remaining is None else remaining
    note = device["note"] if note is None else note
    mode = device.get("mode", "text") if mode is None else mode
    return upsert_device_record(device_id, allowed, remaining, note, mode)


def _touch_device(device_id: str) -> None:
    now = _now()
    with _connect() as conn:
        _exec(
            conn,
            "UPDATE devices SET last_seen_at = %s, updated_at = %s WHERE device_id = %s",
            (now, now, device_id),
        )


def _consume_usage(device_id: str) -> tuple[bool, int, str]:
    now = _now()
    with _connect() as conn:
        cursor = _exec(
            conn,
            """
            UPDATE devices
            SET remaining = remaining - 1, last_seen_at = %s, updated_at = %s
            WHERE device_id = %s AND allowed = TRUE AND remaining > 0
        """,
            (now, now, device_id),
        )
        row = _exec(
            conn,
            "SELECT allowed, remaining FROM devices WHERE device_id = %s",
            (device_id,),
        ).fetchone()
        device = _row_to_dict(row)

    if cursor.rowcount == 1 and device is not None:
        return True, int(device["remaining"]), "allowed"
    if device is None:
        return False, 0, "Device not approved"
    if not bool(device["allowed"]):
        return False, int(device["remaining"]), "Device is not allowed"
    return False, int(device["remaining"]), "Usage limit reached"


def _refund_usage(device_id: str) -> None:
    now = _now()
    with _connect() as conn:
        _exec(
            conn,
            "UPDATE devices SET remaining = remaining + 1, updated_at = %s WHERE device_id = %s",
            (now, device_id),
        )


# ── Event / QA logging (called from background writer thread) ──────────────────
def _do_append_event(
    event: str, device_id: str | None, remaining: int | None, detail: str
) -> None:
    try:
        with _connect() as conn:
            _exec(
                conn,
                "INSERT INTO auth_events (ts, event, device_id, remaining, detail) "
                "VALUES (%s, %s, %s, %s, %s)",
                (_now(), event, device_id, remaining, detail[:1000]),
            )
    except Exception:
        pass


def _do_log_qa(
    device_id: str,
    question: str,
    answer: str,
    option_text: str,
    raw_response: str,
    input_mode: str,
) -> None:
    return
    import sys

    try:
        with _connect() as conn:
            _exec(
                conn,
                "INSERT INTO qa_logs "
                "(ts, device_id, input_mode, question, answer, option_text, raw_response) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    _now(),
                    device_id,
                    (input_mode or "text")[:16],
                    (question or "")[:5000],
                    (answer or "?")[:10],
                    (option_text or "")[:500],
                    (raw_response or "")[:2000],
                ),
            )
        print(
            f"[qa_log] saved — device={device_id} answer={answer}", file=sys.stderr
        )  # ← ADD
    except Exception as exc:  # ← ADD
        print(
            f"[qa_log] FAILED — device={device_id} error={exc}", file=sys.stderr
        )  # ← ADD
        raise


# ── Async fire-and-forget wrappers (enqueue, never block request) ──────────────
def _append_event(
    event: str,
    device_id: str | None = None,
    remaining: int | None = None,
    detail: str = "",
) -> None:
    try:
        _bg_queue.put_nowait(("event", event, device_id, remaining, detail))
    except (asyncio.QueueFull, AttributeError):
        pass


def _log_qa(
    device_id: str,
    question: str,
    answer: str,
    option_text: str,
    raw_response: str,
    input_mode: str = "text",
) -> None:
    try:
        _bg_queue.put_nowait(
            ("qa", device_id, question, answer, option_text, raw_response, input_mode)
        )
    except (asyncio.QueueFull, AttributeError):
        pass


def list_qa_logs(
    limit: int = 100, offset: int = 0, device_id: str | None = None
) -> tuple[list[dict[str, Any]], int]:
    with _connect() as conn:
        if device_id:
            rows = _exec(
                conn,
                "SELECT * FROM qa_logs WHERE device_id = %s "
                "ORDER BY ts DESC LIMIT %s OFFSET %s",
                (device_id, limit, offset),
            ).fetchall()
            total = _exec(
                conn,
                "SELECT COUNT(*) AS c FROM qa_logs WHERE device_id = %s",
                (device_id,),
            ).fetchone()["c"]
        else:
            rows = _exec(
                conn,
                "SELECT * FROM qa_logs ORDER BY ts DESC LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
            total = _exec(conn, "SELECT COUNT(*) AS c FROM qa_logs").fetchone()["c"]
    return [_row_to_dict(r) for r in rows], int(total)


# ── Rate limiting (single atomic Postgres UPSERT — no race condition) ──────────
def _rate_limit_key(request: Request, device_id: str = "") -> str:
    host = request.client.host if request.client else "unknown"
    return f"{host}:{device_id}"


def _check_rate_limit(
    key: str,
    max_requests: int = RATE_LIMIT_MAX_REQUESTS,
    window: int = RATE_LIMIT_WINDOW_SECONDS,
) -> None:
    now = _now()
    window_start = now - (now % window)
    with _connect() as conn:
        cursor = _exec(
            conn,
            """
            INSERT INTO rate_limits (rate_key, window_start, count)
            VALUES (%s, %s, 1)
            ON CONFLICT (rate_key) DO UPDATE
                SET window_start = CASE
                        WHEN rate_limits.window_start <> EXCLUDED.window_start
                        THEN EXCLUDED.window_start
                        ELSE rate_limits.window_start END,
                    count = CASE
                        WHEN rate_limits.window_start <> EXCLUDED.window_start THEN 1
                        WHEN rate_limits.count < %s THEN rate_limits.count + 1
                        ELSE rate_limits.count END
            RETURNING count, window_start
        """,
            (key, window_start, max_requests),
        )
        row = cursor.fetchone()
    if (
        row
        and int(row["count"]) >= max_requests
        and int(row["window_start"]) == window_start
    ):
        raise HTTPException(status_code=429, detail="Too many requests")


# ── Auth helpers ──────────────────────────────────────────────────────────────
def _check_header(request: Request, header_name: str, expected: str) -> None:
    provided = request.headers.get(header_name, "")
    if not expected or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid auth header")


def _token_for(device_id: str) -> str:
    now = _now()
    return jwt.encode(
        {"sub": device_id, "iat": now, "exp": now + JWT_TTL_SECONDS},
        JWT_SECRET,
        algorithm="HS256",
    )


def _admin_token() -> str:
    now = _now()
    return jwt.encode(
        {
            "sub": "admin",
            "scope": "admin",
            "iat": now,
            "exp": now + ADMIN_SESSION_TTL_SECONDS,
        },
        ADMIN_JWT_SECRET,
        algorithm="HS256",
    )


def _decode_bearer_token(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(
            status_code=401, detail="Invalid or expired token"
        ) from None
    device_id = payload.get("sub")
    if not isinstance(device_id, str) or not device_id:
        raise HTTPException(status_code=401, detail="Invalid token subject")
    return device_id


def _require_admin_session(request: Request) -> None:
    token = request.cookies.get("admin_session", "")
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        payload = jwt.decode(token, ADMIN_JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid admin session") from None
    if payload.get("sub") != "admin" or payload.get("scope") != "admin":
        raise HTTPException(status_code=401, detail="Not logged in")


def _require_admin_auth(request: Request) -> None:
    provided_key = request.headers.get(ADMIN_HEADER, "")
    if ADMIN_SHARED_KEY and hmac.compare_digest(provided_key, ADMIN_SHARED_KEY):
        return
    _require_admin_session(request)


# ── Response parsing ──────────────────────────────────────────────────────────
def _response_text(response) -> str:
    # Gemini SDK: response.text
    try:
        return (response.text or "").strip()
    except (AttributeError, ValueError):
        pass
    return ""


def _parse_answer(raw: str) -> dict[str, str]:
    result = {
        "answer": "?",
        "option": "(Option text not detected)",
        "raw": raw,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    for line in raw.splitlines():
        line = line.strip().lstrip("-* ")
        label, sep, value = line.partition(":")
        if not sep:
            continue
        label = label.strip().upper()
        value = value.strip()
        if label == "ANSWER" and value:
            match = re.search(r"\b([A-D])\b", value.upper())
            result["answer"] = match.group(1) if match else value
        elif label == "OPTION" and value:
            result["option"] = value
    return result


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "database": "postgres"}


@app.get("/disclaimer")
def get_disclaimer():
    """Public endpoint — returns the current disclaimer text."""
    return {"disclaimer": _get_setting("disclaimer", "")}


@app.post("/auth")
async def auth(payload: AuthRequest, request: Request):
    _check_header(request, CLIENT_HEADER, CLIENT_SHARED_KEY)
    _check_rate_limit(_rate_limit_key(request, payload.device_id))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    disclaimer = _get_setting("disclaimer", "")

    # Helper — logs PC info in a background thread so it never blocks the response
    def _record_denial(denial_type: str) -> None:
        asyncio.create_task(
            asyncio.to_thread(
                _log_denied_device,
                payload.device_id,
                denial_type,
                payload.pc_name,
                payload.pc_user,
                payload.os_info,
                payload.app_version,
                payload.extra,
            )
        )

    # ── Version gate — reject expired/unversioned clients before any device work ─
    client_version = (payload.app_version or "").strip()
    expired, effective_ver, min_ver = await asyncio.to_thread(
        _version_expired, client_version
    )
    if expired:
        _append_event(
            "denied_version",
            payload.device_id,
            detail=f"client={effective_ver or 'none'} min={min_ver} pc={payload.pc_name or 'unknown'}",
        )
        _record_denial("version_expired")
        return {
            "allowed": False,
            "version_expired": True,
            "device_id": payload.device_id,
            "app_version": effective_ver,
            "min_app_version": min_ver,
            "disclaimer": disclaimer,
            "message": (
                (
                    f"Titan v{effective_ver} is outdated (minimum: v{min_ver}). "
                    "Contact your admin to get the latest version."
                )
                if effective_ver != "0.0.0"
                else (
                    f"This version of Titan is not supported (minimum: v{min_ver}). "
                    "Contact your admin to get the latest version."
                )
            ),
            "timestamp": ts,
        }

    user = await asyncio.to_thread(get_device, payload.device_id)

    if not user:
        _append_event(
            "denied_unknown",
            payload.device_id,
            detail=f"pc={payload.pc_name or 'unknown'}",
        )
        _record_denial("unknown_device")
        return {
            "allowed": False,
            "device_id": payload.device_id,
            "disclaimer": disclaimer,
            "message": "Device not approved. Send this device ID to the admin.",
            "timestamp": ts,
        }

    remaining = int(user.get("remaining", 0))
    if not user.get("allowed", False):
        _append_event(
            "denied_banned",
            payload.device_id,
            remaining,
            detail=f"pc={payload.pc_name or 'unknown'}",
        )
        _record_denial("banned")
        return {
            "allowed": False,
            "device_id": payload.device_id,
            "remaining": remaining,
            "disclaimer": disclaimer,
            "message": "Device is not allowed.",
            "timestamp": ts,
        }

    if remaining <= 0:
        _append_event("denied_limit", payload.device_id, 0)
        return {
            "allowed": False,
            "device_id": payload.device_id,
            "remaining": 0,
            "disclaimer": disclaimer,
            "message": "Usage limit reached.",
            "timestamp": ts,
        }

    await asyncio.to_thread(_touch_device, payload.device_id)
    _append_event("auth_allowed", payload.device_id, remaining)
    return {
        "allowed": True,
        "device_id": payload.device_id,
        "remaining": remaining,
        "token": _token_for(payload.device_id),
        "mode": user.get("mode", "text"),
        "timestamp": ts,
    }


@app.post("/admin/login")
async def admin_login(payload: dict, request: Request, response: Response):
    ip_key = f"adminlogin:{request.client.host if request.client else 'unknown'}"
    _check_rate_limit(
        ip_key, max_requests=ADMIN_LOGIN_RATE_MAX, window=ADMIN_LOGIN_RATE_WINDOW
    )

    password = payload.get("password", "")
    if (
        not ADMIN_PASSWORD
        or not isinstance(password, str)
        or not hmac.compare_digest(password, ADMIN_PASSWORD)
    ):
        raise HTTPException(status_code=401, detail="Invalid password")

    response.set_cookie(
        key="admin_session",
        value=_admin_token(),
        httponly=True,
        secure=ADMIN_COOKIE_SECURE,
        samesite="strict",
        max_age=ADMIN_SESSION_TTL_SECONDS,
    )
    return {"ok": True}


@app.post("/admin/logout")
def admin_logout(response: Response):
    response.delete_cookie(
        "admin_session", secure=ADMIN_COOKIE_SECURE, httponly=True, samesite="strict"
    )
    return {"ok": True}


@app.post("/solve")
async def solve(payload: SolveRequest, request: Request):
    _check_header(request, CLIENT_HEADER, CLIENT_SHARED_KEY)
    device_id = _decode_bearer_token(request)
    _check_rate_limit(_rate_limit_key(request, device_id))

    has_text = bool(payload.text and payload.text.strip())
    has_image = bool(payload.image and payload.image.strip())

    if not has_text and not has_image:
        raise HTTPException(
            status_code=400, detail="Provide at least one of: text, image"
        )

    input_mode = "image" if has_image else "text"

    # Strip data URI prefix — accept raw base64 or data:image/png;base64,... / jpeg
    image_b64: str | None = None
    if has_image:
        raw_image = payload.image.strip()
        if raw_image.startswith("data:"):
            comma_idx = raw_image.find(",")
            if comma_idx == -1:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image data URI: missing comma separator",
                )
            header = raw_image[:comma_idx].lower()
            if (
                "image/png" not in header
                and "image/jpeg" not in header
                and "image/jpg" not in header
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported image type. Only PNG and JPEG accepted.",
                )
            image_b64 = raw_image[comma_idx + 1 :]
        else:
            image_b64 = raw_image
        if not image_b64:
            raise HTTPException(
                status_code=400, detail="Image data is empty after stripping header."
            )

    # Acquire a concurrency slot — runs in thread so event loop stays free during wait
    acquired = await asyncio.to_thread(
        _gemini_slots.acquire, True, GEMINI_QUEUE_TIMEOUT_SECONDS
    )
    if not acquired:
        _append_event("solve_busy", device_id)
        raise HTTPException(
            status_code=429, detail="Server busy — all slots occupied. Try again."
        )

    charged = False
    remaining = 0

    try:
        ok, remaining, message = await asyncio.to_thread(_consume_usage, device_id)
        if not ok:
            _append_event("solve_denied", device_id, remaining, message)
            raise HTTPException(status_code=403, detail=message)
        charged = True

        if input_mode == "image":
            import base64

            image_bytes = base64.b64decode(image_b64)
            gemini_contents = [
                {"mime_type": "image/jpeg", "data": image_bytes},
                "Solve the MCQ visible in this screenshot.",
            ]
            system_prompt = PROMPT_IMAGE
            log_question = "[image-mode]"
        else:
            extracted_text = payload.text.strip()
            gemini_contents = [
                f"Solve this MCQ from the extracted UI text:\n\n{extracted_text}"
            ]
            system_prompt = PROMPT_TEXT
            log_question = extracted_text

        # Blocking Gemini call → thread so ASGI event loop is never stalled
        response = await asyncio.to_thread(
            lambda: _get_gemini(system_prompt).generate_content(
                gemini_contents,
                generation_config=genai.GenerationConfig(temperature=0.1),
            )
        )

        raw = _response_text(response)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if not raw:
            await asyncio.to_thread(_refund_usage, device_id)
            charged = False
            _append_event("solve_empty", device_id, remaining)
            # Return plain-text fallback so payload.cpp can still reload lines
            return PlainTextResponse(
                content="// Empty response from model",
                status_code=200,
            )

        result = _parse_answer(raw)
        result["remaining"] = remaining
        result["input_mode"] = input_mode
        result["timestamp"] = ts

        _append_event("solve_ok", device_id, remaining)
        _log_qa(
            device_id=device_id,
            question=log_question,
            answer=result.get("answer", "?"),
            option_text=result.get("option", ""),
            raw_response=raw,
            input_mode=input_mode,
        )

        # Return raw code as plain text so payload.cpp's ReloadLinesFromCode
        # can split by newline — each line becomes one CTRL+ALT+P keypress.
        return PlainTextResponse(content=raw.strip(), status_code=200)

    except google_exceptions.GoogleAPIError as exc:
        if charged:
            await asyncio.to_thread(_refund_usage, device_id)
        _append_event("solve_gemini_error", device_id, remaining, str(exc))
        raise HTTPException(
            status_code=502, detail=f"Gemini API error: {exc}"
        ) from None
    except HTTPException:
        raise
    except Exception as exc:
        if charged:
            await asyncio.to_thread(_refund_usage, device_id)
        _append_event("solve_error", device_id, remaining, type(exc).__name__)
        raise HTTPException(status_code=502, detail="Model request failed") from None
    finally:
        _gemini_slots.release()


# ── Admin device endpoints ────────────────────────────────────────────────────
@app.get("/admin/devices")
def get_devices(request: Request):
    _require_admin_auth(request)
    return {"devices": list_devices()}


@app.post("/admin/devices")
def upsert_device(payload: DeviceUpsert, request: Request):
    _require_admin_auth(request)
    device = upsert_device_record(
        payload.device_id,
        allowed=payload.allowed,
        remaining=payload.remaining,
        note=payload.note,
        mode=payload.mode,
    )
    _append_event("admin_upsert", payload.device_id, int(device["remaining"]))
    return {"ok": True, "device_id": payload.device_id, "device": device}


@app.patch("/admin/devices/{device_id}")
def patch_device(device_id: str, payload: DevicePatch, request: Request):
    _require_admin_auth(request)
    device = patch_device_record(
        device_id,
        allowed=payload.allowed,
        remaining=payload.remaining,
        note=payload.note,
        mode=payload.mode,
    )
    if device is None:
        raise HTTPException(status_code=404, detail="Device not found")
    _append_event("admin_patch", device_id, int(device["remaining"]))
    return {"ok": True, "device_id": device_id, "device": device}


@app.delete("/admin/devices/{device_id}")
def delete_device(device_id: str, request: Request):
    _require_admin_auth(request)
    with _connect() as conn:
        cursor = _exec(conn, "DELETE FROM devices WHERE device_id = %s", (device_id,))
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Device not found")
    _append_event("admin_delete", device_id)
    return {"ok": True, "device_id": device_id}


# ── Admin app-version settings ────────────────────────────────────────────────
@app.get("/admin/settings/min-version")
def get_min_version(request: Request):
    _require_admin_auth(request)
    min_ver = _get_setting("min_app_version", "1.0.0")
    return {"min_app_version": min_ver}


@app.post("/admin/settings/min-version")
async def set_min_version(payload: dict, request: Request):
    _require_admin_auth(request)
    new_ver = str(payload.get("min_app_version", "")).strip()
    if not new_ver:
        raise HTTPException(status_code=400, detail="min_app_version is required")
    # Basic semver sanity check — must be digits separated by dots
    import re as _re

    if not _re.fullmatch(r"\d+(\.\d+){0,3}", new_ver):
        raise HTTPException(
            status_code=400,
            detail="Invalid version format. Use semver e.g. 1.2.3",
        )
    await asyncio.to_thread(_set_setting, "min_app_version", new_ver)
    _append_event("admin_set_min_version", detail=f"min_app_version={new_ver}")
    return {"ok": True, "min_app_version": new_ver}


@app.get("/admin/settings/require-version")
def get_require_version(request: Request):
    _require_admin_auth(request)
    val = _get_setting("require_version_check", "false").lower() == "true"
    return {"require_version_check": val}


@app.post("/admin/settings/require-version")
async def set_require_version(payload: dict, request: Request):
    _require_admin_auth(request)
    enabled = payload.get("require_version_check")
    if not isinstance(enabled, bool):
        raise HTTPException(
            status_code=400, detail="require_version_check must be a boolean"
        )
    val = "true" if enabled else "false"
    await asyncio.to_thread(_set_setting, "require_version_check", val)
    _append_event("admin_set_require_version", detail=f"require_version_check={val}")
    return {"ok": True, "require_version_check": enabled}


# ── Admin disclaimer settings ─────────────────────────────────────────────────
@app.get("/admin/settings/disclaimer")
def get_disclaimer_setting(request: Request):
    _require_admin_auth(request)
    return {"disclaimer": _get_setting("disclaimer", "")}


@app.post("/admin/settings/disclaimer")
async def set_disclaimer_setting(payload: dict, request: Request):
    _require_admin_auth(request)
    text = str(payload.get("disclaimer", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="disclaimer text is required")
    if len(text) > 2000:
        raise HTTPException(
            status_code=400, detail="disclaimer too long (max 2000 chars)"
        )
    await asyncio.to_thread(_set_setting, "disclaimer", text)
    _append_event("admin_set_disclaimer")
    return {"ok": True, "disclaimer": text}


# ── Admin denied devices log ──────────────────────────────────────────────────
@app.get("/admin/denied-devices")
def get_denied_devices(
    request: Request, limit: int = 100, offset: int = 0, device_id: str = ""
):
    _require_admin_auth(request)
    with _connect() as conn:
        if device_id:
            rows = _exec(
                conn,
                "SELECT * FROM denied_devices WHERE device_id = %s "
                "ORDER BY ts DESC LIMIT %s OFFSET %s",
                (device_id, min(limit, 500), offset),
            ).fetchall()
            total = _exec(
                conn,
                "SELECT COUNT(*) AS c FROM denied_devices WHERE device_id = %s",
                (device_id,),
            ).fetchone()["c"]
        else:
            rows = _exec(
                conn,
                "SELECT * FROM denied_devices ORDER BY ts DESC LIMIT %s OFFSET %s",
                (min(limit, 500), offset),
            ).fetchall()
            total = _exec(conn, "SELECT COUNT(*) AS c FROM denied_devices").fetchone()[
                "c"
            ]
    return {
        "denied": [_row_to_dict(r) for r in rows],
        "total": int(total),
        "count": len(rows),
    }


@app.get("/admin/qa-logs")
def get_qa_logs(
    request: Request, limit: int = 100, offset: int = 0, device_id: str | None = None
):
    _require_admin_auth(request)
    logs, total = list_qa_logs(
        limit=min(limit, 500), offset=offset, device_id=device_id
    )
    return {"logs": logs, "count": len(logs), "total": total}


@app.get("/admin/events")
def get_events(
    request: Request, limit: int = 100, offset: int = 0, device_id: str = ""
):
    _require_admin_auth(request)
    with _connect() as conn:
        if device_id:
            rows = _exec(
                conn,
                "SELECT * FROM auth_events WHERE device_id = %s "
                "ORDER BY ts DESC LIMIT %s OFFSET %s",
                (device_id, limit, offset),
            ).fetchall()
            total = _exec(
                conn,
                "SELECT COUNT(*) AS c FROM auth_events WHERE device_id = %s",
                (device_id,),
            ).fetchone()["c"]
        else:
            rows = _exec(
                conn,
                "SELECT * FROM auth_events ORDER BY ts DESC LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
            total = _exec(conn, "SELECT COUNT(*) AS c FROM auth_events").fetchone()["c"]
    return {"events": [_row_to_dict(r) for r in rows], "total": int(total)}



@app.get("/admin/neo", response_class=HTMLResponse)
def admin_page():
    return """
  <!DOCTYPE html>
    <html lang="en">

    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Nexus Admin | Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #080c14;
                --bg-card: #0e1520;
                --bg-input: #060a10;
                --accent: #4f8ef7;
                --accent2: #7c5cfc;
                --text: #e8edf5;
                --muted: #5a6a82;
                --border: #1a2535;
                --success: #22c55e;
                --danger: #f43f5e;
                --warning: #f59e0b;
                --sidebar-w: 240px;
                --mobile-header-h: 60px;
            }

            *, *::before, *::after {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: 'Space Grotesk', sans-serif;
                background: var(--bg);
                color: var(--text);
                min-height: 100vh;
                overflow-x: hidden;
            }

            /* ── LOGIN ──────────────────────────────────────── */
            #loginBox {
                position: fixed;
                inset: 0;
                z-index: 9999;
                background: var(--bg);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 1.5rem;
            }

            .login-card {
                width: 100%;
                max-width: 400px;
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 2.5rem 2rem;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }

            .login-logo {
                font-size: 2.2rem;
                font-weight: 700;
                background: linear-gradient(135deg, var(--accent), var(--accent2));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.4rem;
            }

            .login-sub {
                color: var(--muted);
                font-size: 0.9rem;
                margin-bottom: 2rem;
            }

            .login-card input {
                width: 100%;
                margin-bottom: 1rem;
                padding: 0.85rem 1rem;
                background: var(--bg-input);
                border: 1px solid var(--border);
                border-radius: 10px;
                color: var(--text);
                font-size: 1rem;
                font-family: 'JetBrains Mono', monospace;
                letter-spacing: 0.1em;
                outline: none;
                transition: border 0.2s;
            }

            .login-card input:focus {
                border-color: var(--accent);
            }

            .btn-primary {
                width: 100%;
                padding: 0.85rem;
                background: linear-gradient(135deg, var(--accent), var(--accent2));
                border: none;
                border-radius: 10px;
                color: #fff;
                font-weight: 600;
                font-size: 1rem;
                cursor: pointer;
                transition: opacity 0.2s, transform 0.15s;
            }

            .btn-primary:hover {
                opacity: 0.9;
                transform: translateY(-2px);
            }

            /* ── SHELL ──────────────────────────────────────── */
            #panel {
                display: none;
            }

            .shell {
                display: flex;
                min-height: 100vh;
                width: 100%;
            }

            /* ── MOBILE HEADER ──────────────────────────────── */
            .mobile-header {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: var(--mobile-header-h);
                background: var(--bg-card);
                border-bottom: 1px solid var(--border);
                z-index: 900;
                align-items: center;
                justify-content: space-between;
                padding: 0 1rem;
            }

            .mobile-header .logo {
                margin: 0;
                padding: 0;
            }

            #hamburger {
                background: transparent;
                border: none;
                color: var(--text);
                font-size: 1.5rem;
                cursor: pointer;
                padding: 0.5rem;
                display: flex;
                align-items: center;
            }

            /* ── SIDEBAR ────────────────────────────────────── */
            #sidebar {
                width: var(--sidebar-w);
                background: var(--bg-card);
                border-right: 1px solid var(--border);
                display: flex;
                flex-direction: column;
                padding: 1.5rem 1rem;
                position: fixed;
                top: 0;
                left: 0;
                bottom: 0;
                z-index: 1000;
                transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            .logo {
                font-size: 1.25rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                background: linear-gradient(135deg, var(--accent), var(--accent2));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 2rem;
                padding-left: 0.5rem;
            }

            .nav-item {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 0.75rem 1rem;
                border-radius: 10px;
                cursor: pointer;
                color: var(--muted);
                font-size: 0.95rem;
                font-weight: 500;
                transition: all 0.2s;
                margin-bottom: 0.5rem;
            }

            .nav-item .icon {
                font-size: 1.1rem;
                width: 24px;
                text-align: center;
            }

            .nav-item:hover {
                color: var(--text);
                background: rgba(79, 142, 247, 0.08);
            }

            .nav-item.active {
                color: var(--accent);
                background: rgba(79, 142, 247, 0.12);
            }

            .nav-bottom {
                margin-top: auto;
            }

            .btn-logout {
                width: 100%;
                padding: 0.75rem;
                background: rgba(244, 63, 94, 0.1);
                border: 1px solid rgba(244, 63, 94, 0.25);
                border-radius: 10px;
                color: var(--danger);
                font-weight: 600;
                font-size: 0.95rem;
                cursor: pointer;
                transition: all 0.2s;
            }

            .btn-logout:hover {
                background: rgba(244, 63, 94, 0.2);
            }

            #overlay {
                display: none;
                position: fixed;
                inset: 0;
                z-index: 950;
                background: rgba(0, 0, 0, 0.6);
                backdrop-filter: blur(2px);
                opacity: 0;
                transition: opacity 0.3s;
            }

            /* ── MAIN ───────────────────────────────────────── */
            #main {
                margin-left: var(--sidebar-w);
                flex: 1;
                padding: 2rem 2.5rem;
                max-width: 100%;
                transition: margin 0.3s;
            }

            .page { display: none; animation: fadeIn 0.3s ease; }
            .page.active { display: block; }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .page-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 2rem;
                gap: 1rem;
                flex-wrap: wrap;
            }

            .page-title {
                font-size: 1.6rem;
                font-weight: 700;
                letter-spacing: -0.03em;
            }

            /* ── STATS ──────────────────────────────────────── */
            .stats-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.25rem;
                margin-bottom: 2rem;
            }

            .stat {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1.5rem;
                transition: transform 0.2s, box-shadow 0.2s;
            }

            .stat:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            }

            .stat-label {
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--muted);
                margin-bottom: 0.5rem;
            }

            .stat-value {
                font-size: 2.2rem;
                font-weight: 700;
                line-height: 1;
            }

            .stat-value.green { color: var(--success); }
            .stat-value.blue { color: var(--accent); }

            /* ── CARDS ──────────────────────────────────────── */
            .card {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 16px;
                margin-bottom: 1.75rem;
                overflow: hidden;
                width: 100%;
            }

            .card-header {
                padding: 1.25rem 1.5rem;
                border-bottom: 1px solid var(--border);
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                flex-wrap: wrap;
            }

            .card-title {
                font-size: 1rem;
                font-weight: 600;
            }

            .card-body { padding: 1.5rem; }

            /* ── FORM ───────────────────────────────────────── */
            .form-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 1.25rem;
                align-items: flex-end;
            }

            .form-grid > *:last-child { /* The save button */
                align-self: flex-end;
            }

            label.field-label {
                display: block;
                font-size: 0.75rem;
                font-weight: 600;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.5rem;
            }

            input[type=text],
            input[type=number],
            input[type=password],
            select {
                width: 100%;
                padding: 0.75rem 1rem;
                background: var(--bg-input);
                border: 1px solid var(--border);
                border-radius: 10px;
                color: var(--text);
                font-family: inherit;
                font-size: 0.95rem;
                outline: none;
                transition: border 0.2s;
            }

            input:focus, select:focus { border-color: var(--accent); }

            .check-row {
                display: flex;
                align-items: center;
                gap: 10px;
                padding-bottom: 8px;
            }

            .check-row input[type=checkbox] {
                width: 18px;
                height: 18px;
                accent-color: var(--accent);
                cursor: pointer;
            }

            .check-row label {
                font-size: 0.95rem;
                cursor: pointer;
            }

            button.btn-save {
                padding: 0.75rem 1.5rem;
                background: linear-gradient(135deg, var(--accent), var(--accent2));
                border: none;
                border-radius: 10px;
                color: #fff;
                font-weight: 600;
                font-size: 0.95rem;
                cursor: pointer;
                white-space: nowrap;
                transition: opacity 0.2s, transform 0.15s;
                height: 42px;
            }

            button.btn-save:hover {
                opacity: 0.9;
                transform: translateY(-1px);
            }

            /* ── SEARCH ─────────────────────────────────────── */
            .search-wrap {
                position: relative;
                flex: 1;
                min-width: 200px;
            }

            .search-wrap input {
                padding-left: 2.5rem;
                height: 40px;
            }

            .search-icon {
                position: absolute;
                left: 1rem;
                top: 50%;
                transform: translateY(-50%);
                color: var(--muted);
                font-size: 1rem;
                pointer-events: none;
            }

            /* ── TABLE ──────────────────────────────────────── */
            .table-wrap {
                width: 100%;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9rem;
                min-width: 800px; /* Forces scrolling on small screens instead of squishing */
            }

            th {
                text-align: left;
                color: var(--muted);
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                padding: 1rem 1.5rem;
                border-bottom: 1px solid var(--border);
                white-space: nowrap;
            }

            td {
                padding: 1rem 1.5rem;
                border-bottom: 1px solid var(--border);
                vertical-align: middle;
            }

            tr:last-child td { border-bottom: none; }
            tr:hover td { background: rgba(255, 255, 255, 0.02); }

            .mono {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
            }

            /* Custom Scrollbar for tables */
            .table-wrap::-webkit-scrollbar { height: 8px; }
            .table-wrap::-webkit-scrollbar-track { background: var(--bg-card); }
            .table-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
            .table-wrap::-webkit-scrollbar-thumb:hover { background: var(--muted); }

            /* ── BADGES ─────────────────────────────────────── */
            .badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 0.7rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .badge-ok { background: rgba(34, 197, 94, 0.15); color: var(--success); }
            .badge-ban { background: rgba(244, 63, 94, 0.15); color: var(--danger); }
            .badge-ans { background: rgba(79, 142, 247, 0.15); color: var(--accent); }

            /* ── ACTION BTNS ────────────────────────────────── */
            .btn-sm {
                padding: 6px 14px;
                border-radius: 8px;
                font-size: 0.8rem;
                font-weight: 600;
                border: none;
                cursor: pointer;
                transition: all 0.2s;
                white-space: nowrap;
            }

            .btn-revoke { background: rgba(244, 63, 94, 0.15); color: var(--danger); }
            .btn-allow { background: rgba(34, 197, 94, 0.15); color: var(--success); }

            .btn-revoke:hover { background: rgba(244, 63, 94, 0.3); }
            .btn-allow:hover { background: rgba(34, 197, 94, 0.3); }

            .btn-refresh {
                padding: 0.6rem 1.25rem;
                border-radius: 10px;
                background: transparent;
                border: 1px solid var(--border);
                color: var(--text);
                font-size: 0.9rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .btn-refresh:hover {
                border-color: var(--accent);
                background: rgba(79, 142, 247, 0.05);
            }

            /* ── LOG ────────────────────────────────────────── */
            pre#log {
                background: #000;
                color: var(--success);
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                padding: 1.25rem;
                border-radius: 10px;
                max-height: 250px;
                overflow: auto;
                white-space: pre-wrap;
            }

            /* ── QA LOG DETAIL ──────────────────────────────── */
            .qa-question {
                max-width: 350px;
                color: var(--muted);
                display: -webkit-box;
                line-clamp: 2;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
                line-height: 1.4;
            }

            /* ── FILTER ROW ─────────────────────────────────── */
            .filter-row {
                display: flex;
                gap: 1rem;
                align-items: center;
                flex-wrap: wrap;
                width: 100%;
                max-width: 500px;
            }

            /* ── PAGINATION ─────────────────────────────────── */
            .pager {
                display: flex;
                gap: 0.75rem;
                align-items: center;
                justify-content: flex-end;
                padding: 1rem 1.5rem;
                font-size: 0.9rem;
                color: var(--muted);
                border-top: 1px solid var(--border);
            }

            .pager button {
                background: var(--bg-input);
                border: 1px solid var(--border);
                border-radius: 8px;
                color: var(--text);
                padding: 6px 16px;
                cursor: pointer;
                font-size: 0.85rem;
                font-weight: 600;
                transition: all 0.2s;
            }

            .pager button:hover:not(:disabled) {
                border-color: var(--accent);
                color: var(--accent);
            }

            .pager button:disabled {
                opacity: 0.4;
                cursor: not-allowed;
            }

            /* ── EMPTY ──────────────────────────────────────── */
            .empty {
                text-align: center;
                padding: 4rem 1rem;
                color: var(--muted);
                font-size: 1rem;
            }

            /* ── UTILS ──────────────────────────────────────── */
            .hidden { display: none !important; }

            /* ── RESPONSIVE MEDIA QUERIES ───────────────────── */

            /* Tablets and smaller laptops */
            @media (max-width: 1024px) {
                #main { padding: 1.5rem; }
                .stats-row { grid-template-columns: repeat(2, 1fr); }
                .filter-row { max-width: 100%; justify-content: flex-end; margin-top: 1rem; }
            }

            /* Mobile phones */
            @media (max-width: 768px) {
                .mobile-header { display: flex; }

                #sidebar {
                    transform: translateX(-100%);
                }

                #sidebar.open {
                    transform: translateX(0);
                    box-shadow: 4px 0 30px rgba(0, 0, 0, 0.8);
                }

                #overlay.open {
                    display: block;
                    opacity: 1;
                }

                #main {
                    margin-left: 0;
                    padding: 1rem;
                    padding-top: calc(var(--mobile-header-h) + 1.5rem);
                }

                .page-header { flex-direction: column; align-items: flex-start; gap: 1rem; }
                .page-title { font-size: 1.4rem; }

                .stats-row { grid-template-columns: 1fr; gap: 1rem; }

                .card-header { flex-direction: column; align-items: flex-start; }
                .filter-row { margin-top: 0.5rem; justify-content: stretch; }
                .filter-row .search-wrap, .filter-row select { width: 100%; flex: none; }

                .form-grid { grid-template-columns: 1fr; }
                .form-grid button.btn-save { width: 100%; height: 48px; margin-top: 0.5rem; }

                th, td { padding: 0.85rem 1rem; }

                /* Version page cards */
                .card[style*="max-width:520px"] { max-width: 100% !important; }
            }
        </style>
    </head>

    <body>

        <div class="mobile-header">
            <div class="logo">◈ Titan</div>
            <button id="hamburger" onclick="toggleSidebar()">☰</button>
        </div>

        <div id="overlay" onclick="closeSidebar()"></div>

        <div id="loginBox">
            <div class="login-card">
                <div class="login-logo">◈ Titan</div>
                <div class="login-sub">Secure Administration Portal</div>
                <input type="password" id="password" placeholder="Access Key" onkeydown="if(event.key==='Enter')login()" />
                <button class="btn-primary" onclick="login()">Authorize Session</button>
                <div id="login-error" style="display:none;margin-top:1rem;color:var(--danger);font-size:0.9rem;font-weight:500;">
                    Incorrect password. Try again.</div>
            </div>
        </div>

        <div id="panel">
            <div class="shell">

                <aside id="sidebar">
                    <div class="logo" style="display: flex; justify-content: space-between; align-items: center;">
                        <span>◈ Titan</span>
                    </div>
                    <nav id="nav">
                        <div class="nav-item active" data-page="dashboard" onclick="switchPage('dashboard')">
                            <span class="icon">⚡</span> Dashboard
                        </div>
                        <div class="nav-item" data-page="devices" onclick="switchPage('devices')">
                            <span class="icon">🖥</span> Devices
                        </div>
                        <div class="nav-item" data-page="qa" onclick="switchPage('qa')">
                            <span class="icon">📋</span> Q&amp;A Logs
                        </div>
                        <div class="nav-item" data-page="events" onclick="switchPage('events')">
                            <span class="icon">📡</span> Auth Events
                        </div>
                        <div class="nav-item" data-page="version" onclick="switchPage('version')">
                            <span class="icon">🔖</span> App Version
                        </div>
                        <div class="nav-item" data-page="denied" onclick="switchPage('denied')">
                            <span class="icon">🚨</span> Denied Devices
                        </div>
                        <div class="nav-item" data-page="disclaimer" onclick="switchPage('disclaimer')">
                            <span class="icon">📝</span> Disclaimer
                        </div>
                    </nav>
                    <div class="nav-bottom">
                        <button class="btn-logout" onclick="logout()">Sign Out</button>
                    </div>
                </aside>

                <main id="main">

                    <div id="page-dashboard" class="page active">
                        <div class="page-header">
                            <div class="page-title">System Overview</div>
                            <button class="btn-refresh" onclick="loadDashboard()">↻ Refresh</button>
                        </div>

                        <div class="stats-row">
                            <div class="stat">
                                <div class="stat-label">Total Devices</div>
                                <div class="stat-value blue" id="st-total">—</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Active</div>
                                <div class="stat-value green" id="st-active">—</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Banned</div>
                                <div class="stat-value" style="color:var(--danger)" id="st-banned">—</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Q&amp;A Logged</div>
                                <div class="stat-value blue" id="st-qa">—</div>
                            </div>
                        </div>

                        <div class="card">
                            <div class="card-header"><span class="card-title">API Console</span></div>
                            <div class="card-body">
                                <pre id="log">> Initializing system console...</pre>
                            </div>
                        </div>
                    </div>

                    <div id="page-devices" class="page">
                        <div class="page-header">
                            <div class="page-title">Device Management</div>
                            <button class="btn-refresh" onclick="loadDevices()">↻ Refresh</button>
                        </div>

                        <div class="card">
                            <div class="card-header"><span class="card-title">Add / Update Device</span></div>
                            <div class="card-body">
                                <div class="form-grid">
                                    <div>
                                        <label class="field-label">Device ID</label>
                                        <input type="text" id="device" placeholder="e.g. USER-99" />
                                    </div>
                                    <div>
                                        <label class="field-label">Usage Limit</label>
                                        <input type="number" id="remaining" placeholder="500" />
                                    </div>
                                    <div>
                                        <label class="field-label">Note (optional)</label>
                                        <input type="text" id="note" placeholder="e.g. test user" />
                                    </div>
                                    <div>
                                        <label class="field-label">Mode</label>
                                        <select id="mode">
                                            <option value="text">📝 Text</option>
                                            <option value="image">🖼 Image</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label class="field-label">Status</label>
                                        <div class="check-row" style="margin-top:12px">
                                            <input type="checkbox" id="allowed" checked />
                                            <label for="allowed">Active</label>
                                        </div>
                                    </div>
                                    <button class="btn-save" onclick="addDevice()">Save</button>
                                </div>
                            </div>
                        </div>

                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">Device Registry</span>
                                <div class="filter-row">
                                    <div class="search-wrap">
                                        <span class="search-icon">🔍</span>
                                        <input type="text" id="device-search" placeholder="Filter devices…"
                                            oninput="filterDevices()" />
                                    </div>
                                </div>
                            </div>
                            <div class="table-wrap">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Device ID</th>
                                            <th>Status</th>
                                            <th>Mode</th>
                                            <th>Remaining</th>
                                            <th>Note</th>
                                            <th>Added</th>
                                            <th>Last Seen</th>
                                            <th>Action</th>
                                        </tr>
                                    </thead>
                                    <tbody id="devices-tbody"></tbody>
                                </table>
                                <div id="devices-empty" class="empty hidden">No devices found.</div>
                            </div>
                        </div>
                    </div>

                    <div id="page-qa" class="page">
                        <div class="page-header">
                            <div class="page-title">Q&amp;A Logs</div>
                            <button class="btn-refresh" onclick="loadQA(true)">↻ Refresh</button>
                        </div>

                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">Recorded Questions &amp; Answers</span>
                                <div class="filter-row">
                                    <div class="search-wrap">
                                        <span class="search-icon">🔍</span>
                                        <input type="text" id="qa-device-filter" placeholder="Filter by device…"
                                            oninput="filterQA()" />
                                    </div>
                                </div>
                            </div>
                            <div class="table-wrap">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Time</th>
                                            <th>Device</th>
                                            <th>Mode</th>
                                            <th>Answer</th>
                                            <th>Option</th>
                                            <th>Question (preview)</th>
                                        </tr>
                                    </thead>
                                    <tbody id="qa-tbody"></tbody>
                                </table>
                                <div id="qa-empty" class="empty hidden">No Q&amp;A logs yet.</div>
                            </div>
                            <div class="pager">
                                <span id="qa-count-label"></span>
                                <button id="qa-prev" onclick="qaPage(-1)" disabled>← Prev</button>
                                <button id="qa-next" onclick="qaPage(1)">Next →</button>
                            </div>
                        </div>
                    </div>

                    <div id="page-events" class="page">
                        <div class="page-header">
                            <div class="page-title">Auth Events</div>
                            <button class="btn-refresh" onclick="loadEvents(true)">↻ Refresh</button>
                        </div>

                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">Recent Auth Activity</span>
                                <div class="filter-row">
                                    <div class="search-wrap">
                                        <span class="search-icon">🔍</span>
                                        <input type="text" id="ev-device-filter" placeholder="Filter by device…"
                                            oninput="filterEvents()" />
                                    </div>
                                    <select id="ev-limit" onchange="loadEvents(true)" style="width:auto; height: 40px; padding: 0 1rem;">
                                        <option value="50">50</option>
                                        <option value="100" selected>100</option>
                                        <option value="250">250</option>
                                    </select>
                                </div>
                            </div>
                            <div class="table-wrap">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Time</th>
                                            <th>Event</th>
                                            <th>Device</th>
                                            <th>Remaining</th>
                                            <th>Detail</th>
                                        </tr>
                                    </thead>
                                    <tbody id="ev-tbody"></tbody>
                                </table>
                                <div id="ev-empty" class="empty hidden">No events found.</div>
                            </div>
                            <div class="pager">
                                <span id="ev-count-label"></span>
                                <button id="ev-prev" onclick="evPage(-1)" disabled>← Prev</button>
                                <button id="ev-next" onclick="evPage(1)">Next →</button>
                            </div>
                        </div>
                    </div>

                    <div id="page-version" class="page">
                        <div class="page-header">
                            <div class="page-title">App Version Control</div>
                            <button class="btn-refresh" onclick="loadVersionSettings()">↻ Refresh</button>
                        </div>

                        <div class="card" style="max-width:520px;margin-bottom:1.2rem">
                            <div class="card-header">
                                <span class="card-title">🔖 Minimum Required Version</span>
                            </div>
                            <div style="padding:1.5rem">
                                <p style="color:var(--muted);font-size:0.95rem;margin-bottom:1.2rem;line-height:1.5">
                                    Clients running a version <strong>below</strong> this value are
                                    rejected at auth and told to contact the admin for an update.
                                </p>
                                <label style="font-size:0.85rem;font-weight:600;color:var(--muted);display:block;margin-bottom:0.4rem;text-transform:uppercase;">Current minimum version</label>
                                <div style="display:flex;gap:0.6rem;align-items:center;margin-bottom:1.5rem">
                                    <span id="cur-min-ver" style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;color:var(--accent)">—</span>
                                </div>
                                <label style="font-size:0.85rem;font-weight:600;color:var(--muted);display:block;margin-bottom:0.4rem;text-transform:uppercase;">Set new minimum version</label>
                                <div style="display:flex;gap:0.75rem;flex-wrap:wrap;">
                                    <input id="new-min-ver" type="text" placeholder="e.g. 1.2.0"
                                        style="flex:1;min-width:150px;padding:0.75rem 1rem;background:var(--bg-input);border:1px solid var(--border);border-radius:10px;color:var(--text);font-family:'JetBrains Mono',monospace;font-size:1rem;outline:none" />
                                    <button onclick="saveMinVersion()"
                                        style="padding:0.75rem 1.5rem;background:linear-gradient(135deg,var(--accent),var(--accent2));border:none;border-radius:10px;color:#fff;font-weight:600;cursor:pointer;white-space:nowrap;transition:opacity 0.2s">
                                        Save
                                    </button>
                                </div>
                                <div id="ver-msg" style="margin-top:1rem;font-size:0.9rem;min-height:1.2em"></div>
                            </div>
                        </div>

                        <div class="card" style="max-width:520px">
                            <div class="card-header">
                                <span class="card-title">🚫 Block Unversioned Clients</span>
                            </div>
                            <div style="padding:1.5rem">
                                <p style="color:var(--muted);font-size:0.95rem;margin-bottom:1.5rem;line-height:1.6">
                                    Old builds of Titan (distributed before version tracking was added)
                                    do <strong>not</strong> send an <code style="background:var(--bg-input);padding:2px 6px;border-radius:4px;font-size:0.85rem;font-family:'JetBrains Mono',monospace;">app_version</code> field.
                                    Enable this to block them — they will see the same
                                    "update required" popup as outdated versioned clients.
                                </p>

                                <div style="display:flex;align-items:center;justify-content:space-between;background:var(--bg-input);border:1px solid var(--border);border-radius:12px;padding:1.25rem;gap:1rem;flex-wrap:wrap;">
                                    <div>
                                        <div style="font-size:1rem;font-weight:600;color:var(--text)" id="req-ver-label">Loading…</div>
                                        <div style="font-size:0.85rem;color:var(--muted);margin-top:4px" id="req-ver-sub"></div>
                                    </div>
                                    <div id="req-ver-toggle" onclick="toggleRequireVersion()"
                                        style="width:52px;height:28px;border-radius:14px;cursor:pointer;transition:background 0.25s;position:relative;flex-shrink:0;background:#333">
                                        <div id="req-ver-knob"
                                            style="width:22px;height:22px;border-radius:50%;background:#fff;position:absolute;top:3px;left:3px;transition:left 0.25s;box-shadow:0 2px 5px rgba(0,0,0,.4)"></div>
                                    </div>
                                </div>
                                <div id="req-ver-msg" style="margin-top:1rem;font-size:0.9rem;min-height:1.2em"></div>
                            </div>
                        </div>
                    </div>

                    <!-- ── DENIED DEVICES PAGE ── -->
                    <div id="page-denied" class="page">
                        <div class="page-header">
                            <div class="page-title">Denied Devices</div>
                            <button class="btn-refresh" onclick="loadDenied(true)">↻ Refresh</button>
                        </div>
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">🚨 Unauthorized Access Attempts</span>
                                <div class="filter-row">
                                    <div class="search-wrap">
                                        <span class="search-icon">🔍</span>
                                        <input type="text" id="denied-device-filter" placeholder="Filter by device…" oninput="loadDenied(true)" />
                                    </div>
                                </div>
                            </div>
                            <div class="table-wrap">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Time</th>
                                            <th>Device ID</th>
                                            <th>Reason</th>
                                            <th>PC Name</th>
                                            <th>PC User</th>
                                            <th>OS</th>
                                            <th>Version</th>
                                            <th>Extra</th>
                                        </tr>
                                    </thead>
                                    <tbody id="denied-tbody"></tbody>
                                </table>
                                <div id="denied-empty" class="empty hidden">No denied device records.</div>
                            </div>
                            <div class="pager">
                                <span id="denied-count-label"></span>
                                <button id="denied-prev" onclick="deniedPage(-1)" disabled>← Prev</button>
                                <button id="denied-next" onclick="deniedPage(1)">Next →</button>
                            </div>
                        </div>
                    </div>

                    <!-- ── DISCLAIMER PAGE ── -->
                    <div id="page-disclaimer" class="page">
                        <div class="page-header">
                            <div class="page-title">Disclaimer Settings</div>
                            <button class="btn-refresh" onclick="loadDisclaimerSetting()">↻ Refresh</button>
                        </div>
                        <div class="card" style="max-width:620px">
                            <div class="card-header">
                                <span class="card-title">📝 Startup Disclaimer Text</span>
                            </div>
                            <div style="padding:1.5rem">
                                <p style="color:var(--muted);font-size:0.95rem;margin-bottom:1.2rem;line-height:1.6">
                                    This text is fetched by the Titan client on every startup and shown to users
                                    who are <strong>denied access</strong> (unknown device or version expired).
                                    It is also fetched on launch so clients always show the latest version.
                                </p>
                                <label style="font-size:0.85rem;font-weight:600;color:var(--muted);display:block;margin-bottom:0.5rem;text-transform:uppercase;">Current Disclaimer</label>
                                <div id="cur-disclaimer" style="background:var(--bg-input);border:1px solid var(--border);border-radius:10px;padding:1rem;font-size:0.9rem;color:var(--text);line-height:1.6;margin-bottom:1.5rem;min-height:60px">Loading…</div>
                                <label style="font-size:0.85rem;font-weight:600;color:var(--muted);display:block;margin-bottom:0.5rem;text-transform:uppercase;">New Disclaimer Text</label>
                                <textarea id="new-disclaimer"
                                    style="width:100%;min-height:120px;padding:0.85rem 1rem;background:var(--bg-input);border:1px solid var(--border);border-radius:10px;color:var(--text);font-family:inherit;font-size:0.95rem;outline:none;resize:vertical;line-height:1.6"
                                    placeholder="Enter disclaimer text…"></textarea>
                                <div style="display:flex;gap:0.75rem;margin-top:1rem;flex-wrap:wrap">
                                    <button onclick="saveDisclaimer()"
                                        style="padding:0.75rem 1.5rem;background:linear-gradient(135deg,var(--accent),var(--accent2));border:none;border-radius:10px;color:#fff;font-weight:600;cursor:pointer;transition:opacity 0.2s">
                                        Save Disclaimer
                                    </button>
                                </div>
                                <div id="disc-msg" style="margin-top:1rem;font-size:0.9rem;min-height:1.2em"></div>
                            </div>
                        </div>
                    </div>

                </main>
            </div></div><script>
            /* ─── STATE ─────────────────────────────────────────── */
            let allDevices = [];
            let allQA = [];
            let allEvents = [];
            let qaOffset = 0;
            const QA_LIMIT = 50;

            /* ─── SIDEBAR ────────────────────────────────────────── */
            function toggleSidebar() {
                document.getElementById('sidebar').classList.toggle('open');
                document.getElementById('overlay').classList.toggle('open');
            }
            function closeSidebar() {
                document.getElementById('sidebar').classList.remove('open');
                document.getElementById('overlay').classList.remove('open');
            }

            /* ─── PAGE SWITCH ───────────────────────────────────── */
            function switchPage(id) {
                document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
                document.getElementById('page-' + id).classList.add('active');
                document.querySelectorAll('.nav-item').forEach(n => {
                    n.classList.toggle('active', n.dataset.page === id);
                });
                closeSidebar();
                if (id === 'devices') loadDevices();
                if (id === 'qa') loadQA(true);
                if (id === 'events') loadEvents(true);
                if (id === 'dashboard') loadDashboard();
                if (id === 'version') loadVersionSettings();
                if (id === 'denied') loadDenied(true);
                if (id === 'disclaimer') loadDisclaimerSetting();
            }

            /* ─── VERSION MANAGEMENT ──────────────────────────────── */
            let _requireVersionEnabled = false;

            async function loadVersionSettings() {
                document.getElementById('cur-min-ver').textContent = '…';
                document.getElementById('req-ver-label').textContent = 'Loading…';
                document.getElementById('req-ver-sub').textContent = '';
                try {
                    const [vRes, rRes] = await Promise.all([
                        fetch('/admin/settings/min-version'),
                        fetch('/admin/settings/require-version')
                    ]);
                    if (!vRes.ok || !rRes.ok) { if (vRes.status === 401 || rRes.status === 401) showLogin(); return; }
                    const vData = await vRes.json();
                    const rData = await rRes.json();
                    document.getElementById('cur-min-ver').textContent = vData.min_app_version || '—';
                    _setRequireToggle(rData.require_version_check === true);
                } catch (e) {
                    document.getElementById('cur-min-ver').textContent = 'Error';
                    document.getElementById('req-ver-label').textContent = 'Error loading';
                }
            }

            function _setRequireToggle(enabled) {
                _requireVersionEnabled = enabled;
                const toggle = document.getElementById('req-ver-toggle');
                const knob   = document.getElementById('req-ver-knob');
                const label  = document.getElementById('req-ver-label');
                const sub    = document.getElementById('req-ver-sub');
                toggle.style.background = enabled ? '#2ecc71' : '#444';
                knob.style.left = enabled ? '27px' : '3px';
                label.textContent = enabled ? 'Blocking unversioned clients' : 'Unversioned clients allowed';
                label.style.color = enabled ? '#2ecc71' : 'var(--text)';
                sub.textContent = enabled
                    ? 'Old builds without app_version will be rejected.'
                    : 'Old builds without app_version can still connect.';
            }

            async function toggleRequireVersion() {
                const newVal = !_requireVersionEnabled;
                const msg = document.getElementById('req-ver-msg');
                msg.textContent = '';
                try {
                    const res = await fetch('/admin/settings/require-version', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ require_version_check: newVal })
                    });
                    const data = await res.json();
                    if (!res.ok) { msg.style.color = '#e74c3c'; msg.textContent = data.detail || 'Error'; return; }
                    _setRequireToggle(data.require_version_check);
                    msg.style.color = '#2ecc71';
                    msg.textContent = newVal ? '✓ Unversioned clients will now be blocked.' : '✓ Unversioned clients will be allowed through.';
                    setTimeout(() => { msg.textContent = ''; }, 3500);
                } catch (e) { msg.style.color = '#e74c3c'; msg.textContent = 'Request failed: ' + e; }
            }

            async function saveMinVersion() {
                const input = document.getElementById('new-min-ver').value.trim();
                const msg = document.getElementById('ver-msg');
                if (!input) { msg.style.color = '#e74c3c'; msg.textContent = 'Please enter a version.'; return; }
                try {
                    const res = await fetch('/admin/settings/min-version', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ min_app_version: input })
                    });
                    const data = await res.json();
                    if (!res.ok) { msg.style.color = '#e74c3c'; msg.textContent = data.detail || 'Error'; return; }
                    msg.style.color = '#2ecc71';
                    msg.textContent = `✓ Minimum version updated to ${data.min_app_version}`;
                    document.getElementById('cur-min-ver').textContent = data.min_app_version;
                    document.getElementById('new-min-ver').value = '';
                    setTimeout(() => { msg.textContent = ''; }, 3500);
                } catch (e) { msg.style.color = '#e74c3c'; msg.textContent = 'Request failed: ' + e; }
            }

            /* ─── LOG ─────────────────────────────────────────────  */
            function log(data) {
                const el = document.getElementById('log');
                el.textContent = `[${new Date().toLocaleTimeString()}]  ` + JSON.stringify(data, null, 2);
            }

            /* ─── AUTH ──────────────────────────────────────────── */
            function showPanel() {
                document.getElementById('loginBox').classList.add('hidden');
                document.getElementById('panel').style.display = 'block';
            }

            async function login() {
                const errEl = document.getElementById('login-error');
                if (errEl) errEl.style.display = 'none';
                try {
                    const res = await fetch('/admin/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ password: document.getElementById('password').value })
                    });
                    if (res.ok) { showPanel(); loadDashboard(); }
                    else {
                        const msg = document.getElementById('login-error');
                        if (msg) { msg.style.display = 'block'; }
                    }
                } catch (e) { log('Login error: ' + e); }
            }

            async function logout() {
                await fetch('/admin/logout', { method: 'POST' }).catch(() => { });
                location.reload();
            }

            /* ─── DASHBOARD ──────────────────────────────────────── */
            async function loadDashboard() {
                try {
                    const [dr, qr] = await Promise.all([
                        fetch('/admin/devices'),
                        fetch('/admin/qa-logs?limit=1')
                    ]);
                    if (!dr.ok) { if (dr.status === 401) showLogin(); return; }
                    const dd = await dr.json();
                    const devs = Object.values(dd.devices || {});
                    document.getElementById('st-total').textContent = devs.length;
                    document.getElementById('st-active').textContent = devs.filter(d => d.allowed).length;
                    document.getElementById('st-banned').textContent = devs.filter(d => !d.allowed).length;
                    if (qr.ok) {
                        const qd = await qr.json();
                        document.getElementById('st-qa').textContent = qd.total ?? qd.count ?? '—';
                    }
                    log({ status: 'ok', devices: devs.length });
                } catch (e) { log('Dashboard error: ' + e); }
            }

            /* ─── DEVICES ─────────────────────────────────────────  */
            async function loadDevices() {
                renderDevices(null);
                try {
                    const res = await fetch('/admin/devices');
                    if (!res.ok) { if (res.status === 401) showLogin(); return; }
                    const data = await res.json();
                    allDevices = Object.values(data.devices || {});
                    renderDevices(allDevices);
                    log(data);
                } catch (e) { log('Load error: ' + e); }
            }

            function filterDevices() {
                const q = document.getElementById('device-search').value.toLowerCase();
                renderDevices(q ? allDevices.filter(d => d.device_id.toLowerCase().includes(q)) : allDevices);
            }

            function renderDevices(list) {
                const tbody = document.getElementById('devices-tbody');
                const empty = document.getElementById('devices-empty');
                if (list === null) { tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:3rem">Loading…</td></tr>'; return; }
                tbody.innerHTML = '';
                if (!list.length) { empty.classList.remove('hidden'); return; }
                empty.classList.add('hidden');
                list.forEach(d => {
                    const date = new Date(d.created_at * 1000).toLocaleDateString();
                    const tr = document.createElement('tr');
                    const lastSeen = d.last_seen_at ? new Date(d.last_seen_at * 1000).toLocaleString() : '—';
                    const mode = d.mode || 'text';

                    const cells = [
                        // Device ID
                        () => { const td = document.createElement('td'); td.className = 'mono'; td.textContent = d.device_id; return td; },
                        // Status
                        () => { const td = document.createElement('td'); const b = document.createElement('span'); b.className = 'badge ' + (d.allowed ? 'badge-ok' : 'badge-ban'); b.textContent = d.allowed ? 'ALLOWED' : 'BANNED'; td.appendChild(b); return td; },
                        // Mode — clickable badge to toggle
                        () => {
                            const td = document.createElement('td');
                            td.style.whiteSpace = 'nowrap';

                            const btn = document.createElement('button');
                            btn.className = 'badge ' + (mode === 'image' ? 'badge-ans' : 'badge-ok');
                            btn.textContent = mode === 'image' ? '🖼 IMAGE' : '📝 TEXT';

                            // Styling to make it look clickable but retain the badge aesthetic
                            btn.style.cursor = 'pointer';
                            btn.style.border = '1px solid transparent';
                            btn.style.transition = 'all 0.2s';
                            btn.title = "Click to toggle mode";

                            btn.onmouseover = () => btn.style.transform = 'translateY(-1px)';
                            btn.onmouseout = () => btn.style.transform = 'translateY(0)';

                            btn.onclick = () => {
                                const newMode = mode === 'image' ? 'text' : 'image';
                                setDeviceMode(d.device_id, newMode);
                            };

                            td.appendChild(btn);
                            return td;
                        },
                        // Remaining
                        () => { const td = document.createElement('td'); td.textContent = '⚡ ' + d.remaining; return td; },
                        // Note
                        () => { const td = document.createElement('td'); td.textContent = d.note || '—'; td.style.color = 'var(--muted)'; td.style.fontSize = '0.85rem'; return td; },
                        // Added
                        () => { const td = document.createElement('td'); td.textContent = date; td.style.color = 'var(--muted)'; return td; },
                        // Last Seen
                        () => { const td = document.createElement('td'); td.textContent = lastSeen; td.style.color = 'var(--muted)'; td.style.fontSize = '0.85rem'; return td; },
                        // Actions
                        () => {
                            const td = document.createElement('td');
                            td.style.display = 'flex'; td.style.gap = '8px';
                            const btn = document.createElement('button');
                            btn.className = 'btn-sm ' + (d.allowed ? 'btn-revoke' : 'btn-allow');
                            btn.textContent = d.allowed ? 'Revoke' : 'Authorize';
                            btn.onclick = () => {
                                if (d.allowed && !confirm('Revoke access for ' + d.device_id + '?')) return;
                                toggleDevice(d.device_id, d.allowed);
                            };
                            const del = document.createElement('button');
                            del.className = 'btn-sm btn-revoke'; del.textContent = 'Delete';
                            del.onclick = () => {
                                if (!confirm('Permanently delete ' + d.device_id + '?')) return;
                                deleteDevice(d.device_id);
                            };
                            td.appendChild(btn); td.appendChild(del); return td;
                        }
                    ];
                    cells.forEach(fn => tr.appendChild(fn()));
                    tbody.appendChild(tr);
                });
            }

            async function addDevice() {
                const device = document.getElementById('device').value.trim();
                const remaining = parseInt(document.getElementById('remaining').value);
                const allowed = document.getElementById('allowed').checked;
                const mode = document.getElementById('mode').value;
                if (!device || isNaN(remaining)) { alert('Invalid inputs'); return; }
                try {
                    const res = await fetch('/admin/devices', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ device_id: device, remaining, allowed, note: document.getElementById('note').value.trim(), mode })
                    });
                    const data = await res.json();
                    log(data);
                    loadDevices();
                } catch (e) { log('Add error: ' + e); }
            }

            async function setDeviceMode(id, newMode) {
                try {
                    const res = await fetch('/admin/devices/' + encodeURIComponent(id), {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ mode: newMode })
                    });
                    const data = await res.json();
                    log(data);
                    // Trigger the popup message on success
                    alert(`Success: Mode for device ${id} has been changed to ${newMode.toUpperCase()}`);
                    loadDevices();
                } catch (e) {
                    log('Mode error: ' + e);
                    alert(`Error: Could not change mode for device ${id}`);
                }
            }

            async function deleteDevice(id) {
                try {
                    const res = await fetch('/admin/devices/' + encodeURIComponent(id), { method: 'DELETE' });
                    const data = await res.json();
                    log(data);
                    loadDevices();
                } catch (e) { log('Delete error: ' + e); }
            }

            async function toggleDevice(id, cur) {
                try {
                    const res = await fetch('/admin/devices/' + encodeURIComponent(id), {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ allowed: !cur })
                    });
                    const data = await res.json();
                    log(data);
                    loadDevices();
                } catch (e) { log('Toggle error: ' + e); }
            }

            /* ─── Q&A LOGS ─────────────────────────────────────────  */
            async function loadQA(reset) {
                if (reset) qaOffset = 0;
                renderQA(null);
                try {
                    const filter = document.getElementById('qa-device-filter').value.trim();
                    let url = `/admin/qa-logs?limit=${QA_LIMIT}&offset=${qaOffset}`;
                    if (filter) url += `&device_id=${encodeURIComponent(filter)}`;
                    const res = await fetch(url);
                    if (!res.ok) { if (res.status === 401) showLogin(); return; }
                    const data = await res.json();
                    allQA = data.logs || [];
                    renderQA(allQA);
                    const label = document.getElementById('qa-count-label');
                    const total = data.total ?? data.count ?? 0;
                    label.textContent = allQA.length ? `Showing ${qaOffset + 1}–${qaOffset + allQA.length} of ${total}` : '';
                    document.getElementById('qa-prev').disabled = qaOffset === 0;
                    document.getElementById('qa-next').disabled = allQA.length < QA_LIMIT;
                } catch (e) { log('QA load error: ' + e); }
            }

            function qaPage(dir) {
                qaOffset = Math.max(0, qaOffset + dir * QA_LIMIT);
                loadQA(false);
            }

            function filterQA() {
                qaOffset = 0;
                loadQA(false);
            }

            function renderQA(list) {
                const tbody = document.getElementById('qa-tbody');
                const empty = document.getElementById('qa-empty');
                if (list === null) { tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--muted);padding:3rem">Loading…</td></tr>'; return; }
                tbody.innerHTML = '';
                if (!list.length) { empty.classList.remove('hidden'); return; }
                empty.classList.add('hidden');
                list.forEach(r => {
                    const tr = document.createElement('tr');
                    const time = new Date(r.ts * 1000).toLocaleString();

                    const tdTime = document.createElement('td'); tdTime.textContent = time; tdTime.style.color = 'var(--muted)'; tdTime.style.whiteSpace = 'nowrap';
                    const tdDev = document.createElement('td'); tdDev.className = 'mono'; tdDev.textContent = r.device_id;

                    const tdMode = document.createElement('td');
                    const modeBadge = document.createElement('span');
                    modeBadge.className = 'badge ' + (r.input_mode === 'image' ? 'badge-ans' : 'badge-ok');
                    modeBadge.textContent = r.input_mode || 'text';
                    tdMode.appendChild(modeBadge);

                    const tdAns = document.createElement('td');
                    const ansBadge = document.createElement('span'); ansBadge.className = 'badge badge-ans'; ansBadge.textContent = r.answer;
                    tdAns.appendChild(ansBadge);

                    const tdOpt = document.createElement('td'); tdOpt.textContent = (r.option_text || '').slice(0, 60) + (r.option_text?.length > 60 ? '…' : '');
                    const tdQ = document.createElement('td');
                    const preview = document.createElement('div'); preview.className = 'qa-question';
                    preview.textContent = r.input_mode === 'image' ? '[image capture]' : (r.question || '').slice(0, 120);
                    tdQ.appendChild(preview);

                    [tdTime, tdDev, tdMode, tdAns, tdOpt, tdQ].forEach(td => tr.appendChild(td));
                    tbody.appendChild(tr);
                });
            }

            function showLogin() {
                document.getElementById('loginBox').classList.remove('hidden');
            }

            /* ─── AUTH EVENTS ───────────────────────────────────── */
            const EV_COLORS = {
                auth_allowed: 'badge-ok',
                denied_unknown: 'badge-ban',
                denied_banned: 'badge-ban',
                denied_limit: 'badge-ban',
                denied_version: 'badge-ban',
                solve_ok: 'badge-ok',
                solve_denied: 'badge-ban',
                solve_error: 'badge-ban',
                solve_gemini_error: 'badge-ban',
                solve_empty: 'badge-ans',
                solve_busy: 'badge-ans',
                admin_upsert: 'badge-ans',
                admin_patch: 'badge-ans',
                admin_delete: 'badge-ban',
                admin_set_min_version: 'badge-ans',
                admin_set_require_version: 'badge-ans',
            };

            let evOffset = 0;
            const EV_LIMIT = 100;

            async function loadEvents(reset) {
                if (reset) evOffset = 0;
                const limit = parseInt(document.getElementById('ev-limit')?.value || EV_LIMIT);
                const device = document.getElementById('ev-device-filter')?.value.trim() || '';
                try {
                    const params = new URLSearchParams({ limit, offset: evOffset });
                    if (device) params.set('device_id', device);
                    const res = await fetch('/admin/events?' + params);
                    if (!res.ok) { if (res.status === 401) showLogin(); return; }
                    const data = await res.json();
                    allEvents = data.events || [];
                    renderEvents(allEvents);
                    const total = data.total ?? allEvents.length;
                    const evLabel = document.getElementById('ev-count-label');
                    if (evLabel) evLabel.textContent = allEvents.length ? `Showing ${evOffset + 1}–${evOffset + allEvents.length} of ${total}` : '';
                    const prevBtn = document.getElementById('ev-prev');
                    const nextBtn = document.getElementById('ev-next');
                    if (prevBtn) prevBtn.disabled = evOffset === 0;
                    if (nextBtn) nextBtn.disabled = allEvents.length < limit;
                } catch (e) { log('Events error: ' + e); }
            }

            function evPage(dir) {
                evOffset = Math.max(0, evOffset + dir * EV_LIMIT);
                loadEvents(false);
            }

            function filterEvents() {
                loadEvents(true);
            }

            function renderEvents(list) {
                const tbody = document.getElementById('ev-tbody');
                const empty = document.getElementById('ev-empty');
                tbody.innerHTML = '';
                if (!list.length) { empty.classList.remove('hidden'); return; }
                empty.classList.add('hidden');
                list.forEach(ev => {
                    const tr = document.createElement('tr');
                    const t = new Date(ev.ts * 1000).toLocaleString();
                    const cls = EV_COLORS[ev.event] || 'badge-ans';

                    const tdTime = document.createElement('td'); tdTime.textContent = t; tdTime.style.whiteSpace = 'nowrap'; tdTime.style.color = 'var(--muted)';
                    const tdEv = document.createElement('td');
                    const badge = document.createElement('span'); badge.className = 'badge ' + cls; badge.textContent = ev.event;
                    tdEv.appendChild(badge);
                    const tdDev = document.createElement('td'); tdDev.className = 'mono'; tdDev.textContent = ev.device_id || '—';
                    const tdRem = document.createElement('td'); tdRem.textContent = ev.remaining ?? '—';
                    const tdDetail = document.createElement('td'); tdDetail.textContent = (ev.detail || '').slice(0, 80); tdDetail.style.color = 'var(--muted)'; tdDetail.style.fontSize = '0.85rem';

                    [tdTime, tdEv, tdDev, tdRem, tdDetail].forEach(td => tr.appendChild(td));
                    tbody.appendChild(tr);
                });
            }

            /* ─── DENIED DEVICES ─────────────────────────────── */
            let deniedOffset = 0;
            const DENIED_LIMIT = 50;

            async function loadDenied(reset) {
                if (reset) deniedOffset = 0;
                const tbody = document.getElementById('denied-tbody');
                const empty = document.getElementById('denied-empty');
                tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:3rem">Loading…</td></tr>';
                empty.classList.add('hidden');
                const filter = document.getElementById('denied-device-filter').value.trim();
                const params = new URLSearchParams({ limit: DENIED_LIMIT, offset: deniedOffset });
                if (filter) params.set('device_id', filter);
                try {
                    const res = await fetch('/admin/denied-devices?' + params);
                    if (!res.ok) { if (res.status === 401) showLogin(); return; }
                    const data = await res.json();
                    const list = data.denied || [];
                    tbody.innerHTML = '';
                    if (!list.length) { empty.classList.remove('hidden'); }
                    const DENIAL_COLORS = {
                        unknown_device: 'badge-ban',
                        version_expired: 'badge-ans',
                        banned: 'badge-ban',
                    };
                    list.forEach(r => {
                        const tr = document.createElement('tr');
                        const cells = [
                            new Date(r.ts * 1000).toLocaleString(),
                            r.device_id || '—',
                            null, // badge
                            r.pc_name || '—',
                            r.pc_user || '—',
                            (r.os_info || '—').slice(0, 40),
                            r.app_version || '—',
                            (r.extra || '—').slice(0, 60),
                        ];
                        cells.forEach((val, i) => {
                            const td = document.createElement('td');
                            if (i === 2) {
                                const b = document.createElement('span');
                                b.className = 'badge ' + (DENIAL_COLORS[r.denial_type] || 'badge-ans');
                                b.textContent = r.denial_type || '—';
                                td.appendChild(b);
                            } else {
                                td.textContent = val;
                                if ([0,5,7].includes(i)) td.style.color = 'var(--muted)';
                                if (i === 1) td.className = 'mono';
                                td.style.fontSize = '0.85rem';
                            }
                            tr.appendChild(td);
                        });
                        tbody.appendChild(tr);
                    });
                    const total = data.total ?? list.length;
                    const label = document.getElementById('denied-count-label');
                    if (label) label.textContent = list.length ? `Showing ${deniedOffset + 1}–${deniedOffset + list.length} of ${total}` : '';
                    document.getElementById('denied-prev').disabled = deniedOffset === 0;
                    document.getElementById('denied-next').disabled = list.length < DENIED_LIMIT;
                } catch (e) { log('Denied load error: ' + e); }
            }

            function deniedPage(dir) {
                deniedOffset = Math.max(0, deniedOffset + dir * DENIED_LIMIT);
                loadDenied(false);
            }

            /* ─── DISCLAIMER ─────────────────────────────────── */
            async function loadDisclaimerSetting() {
                document.getElementById('cur-disclaimer').textContent = 'Loading…';
                try {
                    const res = await fetch('/admin/settings/disclaimer');
                    if (!res.ok) { if (res.status === 401) showLogin(); return; }
                    const data = await res.json();
                    document.getElementById('cur-disclaimer').textContent = data.disclaimer || '(empty)';
                } catch (e) {
                    document.getElementById('cur-disclaimer').textContent = 'Error loading.';
                }
            }

            async function saveDisclaimer() {
                const text = document.getElementById('new-disclaimer').value.trim();
                const msg = document.getElementById('disc-msg');
                if (!text) { msg.style.color = '#e74c3c'; msg.textContent = 'Disclaimer text cannot be empty.'; return; }
                try {
                    const res = await fetch('/admin/settings/disclaimer', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ disclaimer: text })
                    });
                    const data = await res.json();
                    if (!res.ok) { msg.style.color = '#e74c3c'; msg.textContent = data.detail || 'Error'; return; }
                    msg.style.color = '#2ecc71';
                    msg.textContent = '✓ Disclaimer updated successfully.';
                    document.getElementById('cur-disclaimer').textContent = text;
                    document.getElementById('new-disclaimer').value = '';
                    setTimeout(() => { msg.textContent = ''; }, 3500);
                } catch (e) { msg.style.color = '#e74c3c'; msg.textContent = 'Request failed: ' + e; }
            }

            window.onload = async () => {
                try {
                    const res = await fetch('/admin/devices');
                    if (res.ok) { showPanel(); loadDashboard(); }
                } catch { }
            };
        </script>
    </body>

    </html>
    """