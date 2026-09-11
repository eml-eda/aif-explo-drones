from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

CONTROLLERS_DIR = Path(__file__).resolve().parents[2]
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from common.shared_state import (
    CONTROLLER_READY_FILENAME,
    IPC_SESSION_FILENAME,
    SharedStateStore,
    atomic_write_json,
    establish_supervisor_session,
    read_json,
    wait_for_json,
    wait_for_supervisor_session,
)


class SharedStateTests(unittest.TestCase):
    def test_store_resolves_sibling_shared_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller_file = Path(tmp) / "controllers" / "demo_controller" / "demo.py"
            store = SharedStateStore.from_controller_file(controller_file)
            self.assertEqual(store.root, (Path(tmp) / "controllers" / "shared").resolve())
            self.assertTrue(store.root.is_dir())

    def test_atomic_write_and_required_read_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            payload = {"seed": 7, "status": "running"}
            atomic_write_json(path, payload)
            self.assertEqual(read_json(path), payload)
            self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_read_with_default_tolerates_missing_or_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            invalid = Path(tmp) / "invalid.json"
            invalid.write_text("{not-json", encoding="utf-8")
            self.assertEqual(read_json(missing, {"fallback": 1}), {"fallback": 1})
            self.assertEqual(read_json(invalid, []), [])

    def test_required_read_raises_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                read_json(path)

    def test_wait_returns_valid_dictionary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "init.json"
            atomic_write_json(path, {"ready": True, "seed": 3})
            result = wait_for_json(
                path,
                validator=lambda d: d.get("ready") is True,
                timeout_s=0.05,
                poll_s=0.001,
            )
            self.assertEqual(result["seed"], 3)

    def test_wait_timeout_returns_last_dictionary_like_legacy_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "init.json"
            atomic_write_json(path, {"ready": False, "seed": 9})
            result = wait_for_json(
                path,
                validator=lambda d: d.get("ready") is True,
                timeout_s=0.005,
                poll_s=0.001,
            )
            self.assertEqual(result, {"ready": False, "seed": 9})

    def test_store_clear_removes_only_requested_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SharedStateStore(Path(tmp) / "shared")
            store.write("a.json", {"a": 1})
            store.write("b.json", {"b": 2})
            store.write("keep.json", {"keep": True})
            store.clear("a.json", "b.json")
            self.assertFalse(Path(store.path("a.json")).exists())
            self.assertFalse(Path(store.path("b.json")).exists())
            self.assertTrue(Path(store.path("keep.json")).exists())

    def test_store_wait_and_read_use_canonical_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SharedStateStore(Path(tmp) / "shared")
            store.write("task_state.json", {"status": "running"})
            result = store.wait(
                "task_state.json",
                validator=lambda d: d.get("status") == "running",
                timeout_s=0.05,
                poll_s=0.001,
            )
            self.assertEqual(result, store.read("task_state.json"))

    def test_startup_handshake_survives_supervisor_clearing_first_ready_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SharedStateStore(Path(tmp) / "shared")
            token = "fresh-controller-token"
            result_holder = {}
            error_holder = {}

            # Seed stale files from a previous world load.
            store.write(CONTROLLER_READY_FILENAME, {"controller_token": "stale"})
            store.write(IPC_SESSION_FILENAME, {"controller_token": "stale"})

            def controller_side():
                try:
                    result_holder["session"] = wait_for_supervisor_session(
                        store,
                        controller_token=token,
                        timeout_s=0.5,
                        poll_s=0.002,
                    )
                except Exception as exc:  # pragma: no cover - surfaced below
                    error_holder["controller"] = exc

            thread = threading.Thread(target=controller_side)
            thread.start()

            # Let the controller publish at least once, then emulate the
            # supervisor's startup clear. The controller must republish.
            time.sleep(0.01)
            store.clear(CONTROLLER_READY_FILENAME, IPC_SESSION_FILENAME)
            session = establish_supervisor_session(
                store,
                timeout_s=0.5,
                poll_s=0.002,
                environment="hard",
                run_mode="evaluation",
            )
            thread.join(timeout=1.0)

            self.assertFalse(thread.is_alive())
            self.assertNotIn("controller", error_holder)
            self.assertEqual(session["controller_token"], token)
            self.assertEqual(result_holder["session"]["controller_token"], token)
            self.assertEqual(result_holder["session"]["environment"], "hard")

    def test_controller_handshake_rejects_stale_session_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SharedStateStore(Path(tmp) / "shared")
            store.write(IPC_SESSION_FILENAME, {"controller_token": "previous-run"})
            with self.assertRaises(TimeoutError):
                wait_for_supervisor_session(
                    store,
                    controller_token="current-run",
                    timeout_s=0.01,
                    poll_s=0.001,
                )


if __name__ == "__main__":
    unittest.main()
