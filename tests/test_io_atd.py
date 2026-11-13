"""Synthetic tests for the ATD loader. They never touch the match data.

    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import isolation  # noqa: E402,F401  (must come before possession imports)

import baseline  # noqa: E402
from possession import baseline_check, io_atd  # noqa: E402

META = """<?xml version="1.0"?>
<Main><Metadata>
  <Teams><Team id="T1"><Name>Home FC</Name></Team><Team id="T2"><Name>Away FC</Name></Team></Teams>
  <Players>
    <Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/>
    <Player id="P3" teamId="T1"/><Player id="P4" teamId="T2"/>
  </Players>
</Metadata></Main>
"""

NAN = "NaN,NaN,NaN"


def record(frame_id, players, ball):
    return "{}:{}:{}\n".format(frame_id, ";".join(players), ball)


class SyntheticATD(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        folder = os.path.join(self.tmp.name, "tactical")
        os.makedirs(folder)
        self.meta = os.path.join(folder, "BAR-ATH_tactical_atd.xml")
        self.raw = os.path.join(folder, "BAR-ATH_tactical_atd.txt")
        with open(self.meta, "w") as handle:
            handle.write(META)
        lines = [
            # ball next to slot 1 (home)
            record(1, ["0.5,0.5,1.0", "0.9,0.9,2.0", "0.1,0.1,0", "0.2,0.8,0"], "0.51,0.5,3.0"),
            # ball next to slot 2 (away)
            record(2, ["0.5,0.5,1.0", "0.9,0.9,2.0", "0.1,0.1,0", NAN], "0.89,0.9,3.0"),
            # ball missing
            record(3, ["0.5,0.5,1.0", "0.9,0.9,2.0", "0.1,0.1,0", NAN], "NaN,NaN,NaN"),
            # ball near a missing slot: the nearest *present* slot wins (slot 3, home)
            record(4, [NAN, NAN, "0.1,0.1,0", "0.9,0.1,0"], "0.15,0.12,0.0"),
            # exact tie between slot 1 and slot 2: first slot wins in both implementations
            record(6, ["0.4,0.5,0", "0.6,0.5,0", NAN, NAN], "0.5,0.5,0"),
        ]
        with open(self.raw, "w") as handle:
            handle.writelines(lines)
        self.arrays = io_atd.build(data_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_shapes_and_alignment(self):
        self.assertEqual(self.arrays["slots"].shape, (199319, 4, 3))
        self.assertEqual(self.arrays["ball"].shape, (199319, 3))
        # frame_id 1 -> video frame 0; frame_id 5 absent -> row 4 not present
        self.assertTrue(self.arrays["present"][[0, 1, 2, 3, 5]].all())
        self.assertFalse(self.arrays["present"][4])
        self.assertAlmostEqual(self.arrays["ball"][0, 0], 0.51)
        np.testing.assert_array_equal(self.arrays["team"], [0, 1, 0, 1])

    def test_matches_reference_baseline(self):
        reference = baseline.nearest_player_labels(
            self.raw, baseline.team_of_each_track(self.meta), 199319, (105.3, 68.0))
        ours = baseline_check.nearest_player_codes(self.arrays)
        np.testing.assert_array_equal(ours, reference)
        self.assertEqual(list(ours[:6]), [baseline.HOME, baseline.AWAY, baseline.NOBODY,
                                          baseline.HOME, baseline.NOBODY, baseline.HOME])

    def test_rejects_inconsistent_record(self):
        with open(self.raw, "a") as handle:
            handle.write(record(7, ["0.5,0.5,1.0"], "0.5,0.5,0"))
        with self.assertRaises(ValueError):
            io_atd.build(data_dir=self.tmp.name)


if __name__ == "__main__":
    unittest.main()
