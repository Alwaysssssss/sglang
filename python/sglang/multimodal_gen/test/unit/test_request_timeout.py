import importlib.util
import tempfile
import types
import unittest
from pathlib import Path


_REQUEST_TIMEOUT_PATH = (
    Path(__file__).resolve().parents[2] / "runtime" / "request_timeout.py"
)
_SPEC = importlib.util.spec_from_file_location("request_timeout", _REQUEST_TIMEOUT_PATH)
request_timeout = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(request_timeout)

TaskTimeoutError = request_timeout.TaskTimeoutError
check_request_timeout = request_timeout.check_request_timeout


class TestRequestTimeout(unittest.TestCase):
    def test_cancel_marker_raises_task_timeout(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        cancel_path = Path(temp_dir.name) / "request.cancel"
        cancel_path.write_text("cancel")

        request = types.SimpleNamespace(
            sampling_params=types.SimpleNamespace(request_cancel_path=str(cancel_path))
        )

        with self.assertRaises(TaskTimeoutError):
            check_request_timeout(request)

    def test_missing_cancel_marker_does_not_raise(self):
        request = types.SimpleNamespace(
            sampling_params=types.SimpleNamespace(request_cancel_path="missing.cancel")
        )

        check_request_timeout(request)


if __name__ == "__main__":
    unittest.main()
