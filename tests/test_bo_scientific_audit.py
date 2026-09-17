import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bo_scientific_audit import audit


class ScientificAuditTests(unittest.TestCase):
    def test_audit_stops_without_fully_matched_bo_region(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results.json"
            source.write_text(json.dumps({"records": [{
                "checkpoint_id": "models/run",
                "fully_matched": True,
                "metrics": {"direct_bo_candidate": {"mean": 0.0}, "linear_bo_candidate": {"mean": 0.0}},
            }]}), encoding="utf-8")
            report, json_path, csv_path, markdown_path = audit([source], root / "audit")
            self.assertEqual(report["recommendation"], "STOP_AND_REVISE_BO_DEFINITION")
            self.assertTrue(json_path.is_file() and csv_path.is_file() and markdown_path.is_file())

    def test_audit_proceeds_with_nonempty_fully_matched_region(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results.json"
            source.write_text(json.dumps({"records": [{
                "checkpoint_id": "models/run",
                "fully_matched": True,
                "metrics": {"direct_bo_candidate": {"mean": 0.25}, "linear_bo_candidate": {"mean": 0.0}},
            }]}), encoding="utf-8")
            report, *_ = audit([source], root / "audit")
            self.assertEqual(report["recommendation"], "PROCEED_MAIN_BO")


if __name__ == "__main__":
    unittest.main()
