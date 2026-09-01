import importlib.util
import json
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "scripts" / "export_streamlit_dashboard.py"


class StreamlitReadonlySnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SIMLINGO_DASHBOARD_READ_ONLY"] = "1"
        spec = importlib.util.spec_from_file_location("readonly_exporter", EXPORTER)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {EXPORTER}")
        cls.exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.exporter)

    def test_snapshot_rejects_non_get_api_calls(self):
        dashboard = self.exporter.load_dashboard_module()
        html, snapshot, _, route_count = self.exporter.build_snapshot(dashboard)
        self.assertTrue(snapshot["/api/config"]["read_only"])
        self.assertGreaterEqual(route_count, 200)
        self.assertIn('method !== "GET"', html)
        self.assertNotIn("const r = await fetch(path, opts);", html)

    def test_committed_kpis_are_sanitized_and_present(self):
        payload = json.loads(
            (ROOT / "streamlit_share" / "kpi_snapshot.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["schema"], "vla_av_readonly_kpi_snapshot_v1")
        self.assertTrue(payload["comparison"]["cards"])
        serialized = json.dumps(payload)
        self.assertNotIn(str(ROOT), serialized)
        self.assertNotIn(str(Path.home()), serialized)

    def test_streamlit_host_has_no_backend_process_api(self):
        source = (ROOT / "streamlit_share" / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("subprocess", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("/api/launch", source)


if __name__ == "__main__":
    unittest.main()
