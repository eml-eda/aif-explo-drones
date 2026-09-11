from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_MISSING = object()


def read_json(path: str | os.PathLike[str], default: Any = _MISSING) -> Any:
    """Read JSON from *path*.

    If ``default`` is supplied, missing/invalid files return that value. This
    preserves the controller/supervisor polling semantics used by the project.
    If ``default`` is omitted, the underlying file/JSON exception is allowed to
    propagate so required files fail loudly.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        if default is _MISSING:
            raise
        return default


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Any,
    *,
    indent: int = 2,
) -> None:
    """Atomically replace a JSON file using the project's existing ``.tmp`` convention."""
    path_str = os.fspath(path)
    parent = os.path.dirname(path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_path = path_str + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)
    os.replace(tmp_path, path_str)


def safe_remove(path: str | os.PathLike[str]) -> None:
    """Best-effort removal for transient shared-state files."""
    try:
        path_str = os.fspath(path)
        if os.path.exists(path_str):
            os.remove(path_str)
    except Exception:
        pass


def wait_for_json(
    path: str | os.PathLike[str],
    validator: Callable[[dict], bool],
    timeout_s: float,
    poll_s: float,
):
    """Poll a JSON file until ``validator`` accepts it or timeout expires.

    The timeout return value intentionally matches the pre-refactor behavior:
    the most recent dictionary is returned even when it never validated;
    ``None`` is returned if no dictionary was observed.
    """
    t0 = time.time()
    last_data = None
    while (time.time() - t0) < float(timeout_s):
        data = read_json(path, None)
        if isinstance(data, dict):
            last_data = data
            try:
                if validator(data):
                    return data
            except Exception:
                pass
        time.sleep(float(poll_s))
    return last_data


@dataclass(frozen=True)
class SharedStateStore:
    """Canonical access point for ``webots/controllers/shared`` runtime files."""

    root: Path

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "root", root)

    @classmethod
    def from_controller_file(cls, controller_file: str | os.PathLike[str]) -> "SharedStateStore":
        controller_path = Path(controller_file).resolve()
        controllers_dir = controller_path.parent.parent
        return cls(controllers_dir / "shared")

    def path(self, filename: str) -> str:
        return str(self.root / filename)

    def read(self, filename: str, default: Any = _MISSING) -> Any:
        return read_json(self.root / filename, default)

    def write(self, filename: str, payload: Any, *, indent: int = 2) -> None:
        atomic_write_json(self.root / filename, payload, indent=indent)

    def wait(
        self,
        filename: str,
        validator: Callable[[dict], bool],
        timeout_s: float,
        poll_s: float,
    ):
        return wait_for_json(
            self.root / filename,
            validator=validator,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )

    def remove(self, filename: str) -> None:
        safe_remove(self.root / filename)

    def clear(self, *filenames: str) -> None:
        for filename in filenames:
            self.remove(filename)

CONTROLLER_READY_FILENAME = "controller_ready.json"
IPC_SESSION_FILENAME = "ipc_session.json"


def wait_for_supervisor_session(
    store: SharedStateStore,
    *,
    controller_token: str,
    timeout_s: float = 10.0,
    poll_s: float = 0.05,
) -> dict:
    """Establish a fresh controller->supervisor startup handshake.

    The controller repeatedly rewrites ``controller_ready.json`` until the
    supervisor acknowledges the same token in ``ipc_session.json``. Rewriting
    is intentional: if the supervisor starts second it may clear the first
    ready file while removing stale state from the previous episode.
    """
    import uuid

    token = str(controller_token).strip() or uuid.uuid4().hex
    deadline = time.time() + float(timeout_s)
    ready_payload = {
        "controller_token": token,
        "controller_pid": int(os.getpid()),
    }

    while time.time() < deadline:
        ready_payload["updated_at_unix_s"] = float(time.time())
        store.write(CONTROLLER_READY_FILENAME, ready_payload)

        session = store.read(IPC_SESSION_FILENAME, None)
        if (
            isinstance(session, dict)
            and str(session.get("controller_token", "")) == token
        ):
            return session

        time.sleep(float(poll_s))

    raise TimeoutError(
        "Timed out waiting for supervisor IPC session acknowledgement "
        f"for controller token {token!r}."
    )


def establish_supervisor_session(
    store: SharedStateStore,
    *,
    timeout_s: float = 10.0,
    poll_s: float = 0.05,
    environment: str | None = None,
    run_mode: str | None = None,
) -> dict:
    """Wait for the current controller process and acknowledge its token.

    Call this only *after* clearing ``controller_ready.json`` and
    ``ipc_session.json``. That ordering guarantees that the acknowledged token
    came from the controller process participating in the current world load,
    rather than from a previous episode.
    """
    ready = store.wait(
        CONTROLLER_READY_FILENAME,
        validator=lambda d: bool(str(d.get("controller_token", "")).strip()),
        timeout_s=timeout_s,
        poll_s=poll_s,
    )
    token = "" if not isinstance(ready, dict) else str(ready.get("controller_token", "")).strip()
    if not token:
        raise TimeoutError("Timed out waiting for the controller startup handshake.")

    session = {
        "controller_token": token,
        "supervisor_pid": int(os.getpid()),
        "established_at_unix_s": float(time.time()),
    }
    if environment is not None:
        session["environment"] = str(environment)
    if run_mode is not None:
        session["run_mode"] = str(run_mode)

    store.write(IPC_SESSION_FILENAME, session)
    return session
