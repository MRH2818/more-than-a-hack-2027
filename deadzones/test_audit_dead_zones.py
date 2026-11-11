import unittest

from audit_dead_zones import audit, verified_runs
from dead_zones import Config, Event, dead_mask, infer


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            Event("p", 1, 4, "PROGRESSION", "1st half", "Barcelona", ""),
            Event("r", 10, 45, "SET PIECES", "1st half", "N/A", "throw-in"),
        ]
        self.analysis = infer(self.events, Config(fps=25, n_frames=1250))

    def test_xml_alone_never_asserts_certain_dead_or_live(self):
        result = audit(self.events, self.analysis)
        self.assertEqual({r["status"] for r in result["frames"]}, {"needs_evidence"})
        self.assertEqual(result["summary"]["gap_frames"], 100)
        self.assertEqual(result["summary"]["gap_certain_dead_frames"], 0)
        self.assertFalse(verified_runs(result["frames"]))

    def test_reference_splits_frames_and_does_not_modify_predictions(self):
        labels = ["X"] * 1250
        labels[150:160] = ["D"] * 10
        labels[160:165] = ["A"] * 5
        before = dead_mask(self.analysis, "exploratory")
        result = audit(self.events, self.analysis, labels, "test_reference.csv")
        self.assertEqual(verified_runs(result["frames"]), [[150, 160]])
        rows = {r["frame"]: r for r in result["frames"]}
        self.assertEqual(rows[149]["status"], "needs_evidence")
        self.assertTrue(rows[150]["certain_dead"])
        self.assertEqual(rows[160]["status"], "reference_live")
        self.assertFalse(rows[160]["certain_dead"])
        self.assertEqual(rows[165]["status"], "needs_evidence")
        self.assertEqual(result["gaps"][0]["certain_dead_frames"], 10)
        self.assertEqual(result["gaps"][0]["known_live_frames"], 5)
        self.assertEqual(dead_mask(self.analysis, "exploratory"), before)
        self.assertFalse(any(dead_mask(self.analysis, "strict")))

    def test_overlapping_windows_are_not_double_counted(self):
        other = Event("r2", 20, 50, "SET PIECES", "1st half", "N/A", "goal kick")
        events = self.events + [other]
        result = audit(events, infer(events, Config(fps=25, n_frames=1250)))
        self.assertEqual(result["summary"]["unique_review_frames"], (50 - 4) * 25)
        shared = next(r for r in result["frames"] if r["frame"] == 600)
        self.assertEqual(shared["restart_tag_ids"], ["r", "r2"])

    def test_play_tag_conflict_is_not_verified_live(self):
        phase = Event("p2", 12, 18, "BUILD UP", "1st half", "Barcelona", "")
        events = self.events + [phase]
        result = audit(events, infer(events, Config(fps=25, n_frames=1250)))
        row = next(r for r in result["frames"] if r["frame"] == 350)
        self.assertEqual(row["status"], "needs_evidence")
        self.assertIn("resolve_overlapping_play_tag", row["evidence_needed"])

    def test_segments_reconstruct_frame_rows_exactly(self):
        result = audit(self.events, self.analysis)
        expanded = []
        for s in result["segments"]:
            evidence = {k: v for k, v in s.items() if k not in {"start_frame", "end_frame_exclusive"}}
            expanded.extend({"frame": i, **evidence}
                            for i in range(s["start_frame"], s["end_frame_exclusive"]))
        self.assertEqual(expanded, result["frames"])
        self.assertEqual(result["frames"][0]["frame"], 100)
        self.assertEqual(result["frames"][-1]["frame"], 1124)

    def test_reference_requires_alignment_and_provenance(self):
        with self.assertRaises(ValueError):
            audit(self.events, self.analysis, ["D"], "ref")
        with self.assertRaises(ValueError):
            audit(self.events, self.analysis, ["D"] * 1250)

    def test_reference_outside_review_scope_is_not_reported(self):
        labels = ["X"] * 1250
        labels[:100] = ["D"] * 100
        result = audit(self.events, self.analysis, labels, "ref")
        self.assertFalse(verified_runs(result["frames"]))


if __name__ == "__main__":
    unittest.main()
