import tempfile
import unittest
from pathlib import Path

from deadzones.dead_zones import (Config, Event, dead_mask, evaluate_public, infer,
                        load_events, overlay_labels, read_labels)


def event(id, start, end, code, kind="", half="1st half"):
    return Event(str(id), start, end, code, half, "Barcelona", kind)


class DeadZonesTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(fps=25, n_frames=5000)
        self.events = [
            event(1, 10, 81, "GOALS"),
            event(2, 80, 85, "SET PIECES", "kick off (after goal)"),
            event(3, 10, 30, "BALL IN THE BOX"),
            event(4, 20, 40, "SET PIECES", "corner kick"),
        ]

    def test_goal_clip_overlap_does_not_remove_pair_or_include_buildup(self):
        result = infer(self.events, self.config)
        mask = dead_mask(result)
        self.assertEqual(len(result["pairs"]), 1)
        self.assertFalse(any(mask[:1025]))  # internal corner clip ends at 40s + 1s guard
        self.assertTrue(all(mask[1025:1975]))
        self.assertFalse(mask[1975])        # restart start - guard, exclusive
        self.assertEqual(sum(mask), 950)

    def test_phase_spanning_restart_cancels_goal_core(self):
        events = self.events + [event(5, 60, 84, "PROGRESSION")]
        result = infer(events, self.config)
        self.assertFalse(any(dead_mask(result)))
        self.assertTrue(any("No positive core" in x["reason"] for x in result["issues"]))

    def test_goal_without_play_evidence_abstains(self):
        self.assertFalse(any(dead_mask(infer(self.events[:2], self.config))))

    def test_do_not_pair_across_halves(self):
        events = [self.events[0], event(2, 80, 85, "SET PIECES",
                  "kick off (after goal)", "2nd half"), self.events[2]]
        self.assertFalse(any(dead_mask(infer(events, self.config))))

    def test_missing_half_abstains(self):
        events = [event(1, 10, 81, "GOALS", half=""),
                  event(2, 80, 85, "SET PIECES", "kick off (after goal)", half=""),
                  event(3, 10, 30, "BALL IN THE BOX", half="")]
        self.assertFalse(any(dead_mask(infer(events, self.config))))

    def test_ambiguous_goals_cannot_reuse_kickoff(self):
        result = infer(self.events + [event(5, 15, 81, "GOALS")], self.config)
        self.assertFalse(any(dead_mask(result)))
        self.assertTrue(any("Ambiguous" in x["reason"] for x in result["issues"]))

    def test_intervening_restart_blocks_a_distant_kickoff(self):
        events = [event(1, 10, 35, "GOALS"), self.events[1], self.events[2],
                  event(5, 45, 50, "SET PIECES", "throw-in")]
        result = infer(events, self.config)
        self.assertFalse(any(dead_mask(result)))
        self.assertTrue(any("Intervening" in x["reason"] for x in result["issues"]))

    def test_goal_span_limit(self):
        result = infer(self.events, Config(n_frames=5000, max_goal_span_s=50))
        self.assertFalse(any(dead_mask(result)))

    def test_weak_gap_is_review_only(self):
        events = [event(1, 10, 20, "PROGRESSION"),
                  event(2, 30, 65, "SET PIECES", "throw-in")]
        result = infer(events, self.config)
        self.assertFalse(any(dead_mask(result)))
        self.assertEqual(sum(dead_mask(result, "exploratory")), 200)
        self.assertFalse(any(dead_mask(result, "strict")))
        # A different tactical clip crossing the gap invalidates this candidate.
        result = infer(events + [event(3, 19, 40, "BUILD UP")], self.config)
        self.assertFalse(any(dead_mask(result, "exploratory")))

    def test_overlay_preserves_unknowns_and_existing_predictions(self):
        result = infer(self.events, self.config)
        labels = ["H", "A", "X", "D"] * 1250
        original = labels[:]
        output = overlay_labels(labels, result)
        self.assertEqual(labels, original)
        self.assertEqual(output[:1025], labels[:1025])
        self.assertEqual(set(output[1025:1975]), {"D"})
        self.assertEqual(output[1975:], labels[1975:])
        self.assertEqual(overlay_labels(labels, result, "strict"), labels)
        with self.assertRaises(ValueError):
            overlay_labels(labels[:-1], result)

    def test_ground_truth_unknowns_are_not_treated_as_live(self):
        result = infer(self.events, self.config)
        labels = ["X"] * 5000
        labels[1025:1030] = ["D"] * 5
        labels[1030:1033] = ["H"] * 3
        stats = evaluate_public(result, labels)["goals"]
        self.assertEqual(stats["graded_predicted_dead"], 8)
        self.assertEqual(stats["precision"], 5 / 8)
        self.assertEqual(stats["recall"], 1)
        self.assertIsNone(evaluate_public(result, ["X"] * 5000)["goals"]["precision"])

    def test_input_order_does_not_change_frames(self):
        self.assertEqual(dead_mask(infer(self.events, self.config)),
                         dead_mask(infer(list(reversed(self.events)), self.config)))

    def test_clipping_to_video_length_and_mask_union(self):
        result = infer(self.events, Config(n_frames=1500))
        self.assertEqual(len(dead_mask(result)), 1500)
        self.assertEqual(sum(dead_mask(result)), 475)
        result["pairs"] *= 2
        self.assertEqual(sum(dead_mask(result)), 475)  # never double-count overlaps

    def test_config_rejects_invalid_numbers(self):
        for kwargs in ({"fps": 0}, {"guard_s": -1}, {"guard_s": float("nan")},
                       {"n_frames": 0}, {"max_restart_gap_s": float("inf")}):
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_xml_validation_and_normalization(self):
        xml = """<file><ALL_INSTANCES><instance><ID>1</ID><start>1</start><end>3</end>
        <code>set pieces</code><label><group>Type</group><text>Kick Off (After Goal)</text></label>
        <label><group>Half</group><text>1st Half</text></label></instance></ALL_INSTANCES></file>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.xml"
            path.write_text(xml)
            self.assertEqual(load_events(path)[0].kind, "kick off (after goal)")
            for invalid in (xml.replace("<start>1", "<start>nan"),
                            xml.replace("<end>3", "<end>0"),
                            xml.replace("</ALL_INSTANCES>",
                                        xml.split("<ALL_INSTANCES>")[1].split("</ALL_INSTANCES>")[0]
                                        + "</ALL_INSTANCES>")):
                path.write_text(invalid)
                with self.assertRaises(ValueError):
                    load_events(path)

    def test_csv_requires_zero_based_consecutive_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text("frame,label\n1,H\n2,A\n")
            with self.assertRaises(ValueError):
                read_labels(path, 2)

    def test_delivered_file_has_unique_pairs_and_no_tag_overlap_in_cores(self):
        path = Path(__file__).resolve().parents[1] / "data/tactical/BAR-ATH_tactical_smart_tagging.xml"
        if not path.exists():
            self.skipTest("Gated match XML not present")
        events = load_events(path)
        result = infer(events)
        goals = [p for p in result["pairs"] if p["rule"] == "goal_to_kickoff"]
        self.assertEqual(len(events), 638)
        self.assertEqual(len(goals), 4)
        self.assertEqual(len({p["restart_event_id"] for p in goals}), 4)
        self.assertEqual(len(result["restart_review"]), 59)
        for p in goals:
            overlaps = [e.id for e in events if e.code != "GOALS"
                        and e.start < p["end_s"] and e.end > p["start_s"]]
            self.assertEqual(overlaps, [])


if __name__ == "__main__":
    unittest.main()
