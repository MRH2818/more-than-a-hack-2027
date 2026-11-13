"""Synthetic tests for the whistle detector and its decoder bonus. Never reads cache/ or data/."""

import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import isolation  # noqa: E402,F401  (must come before possession imports)

from possession import audio, config, decoder, pipeline  # noqa: E402
from test_pipeline import kickoff_events, synthetic_match  # noqa: E402

RATE = config.AUDIO_RATE


def crowd_with_whistles(seconds=60, whistles=((10.0, 0.5), (30.0, 1.0), (45.0, 0.3)),
                        tone_hz=3150.0, seed=0):
    """Broadband crowd noise plus steady whistle tones at (start s, length s)."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(seconds * RATE)) / float(RATE)
    signal = 0.2 * rng.randn(len(t))
    for start, length in whistles:
        on = (t >= start) & (t < start + length)
        signal[on] += 0.3 * np.sin(2 * np.pi * tone_hz * t[on])
    return signal.astype(np.float32)


class WhistleTests(unittest.TestCase):
    def test_finds_whistles_at_the_right_frames(self):
        sound = crowd_with_whistles()
        features = audio.frame_features(sound, 0.0, 60 * 25)
        events = audio.whistles(features)
        self.assertEqual(len(events), 3)
        for (start, stop), (when, length) in zip(events, ((10.0, 0.5), (30.0, 1.0), (45.0, 0.3))):
            self.assertLessEqual(abs(start - when * 25), 2)
            self.assertLessEqual(abs((stop - start) - length * 25), 3)
        self.assertTrue(np.all(np.abs(features["peak_hz"][events[0][0] + 2:events[0][1] - 2]
                                      - 3150) < 40))

    def test_noise_alone_gives_no_whistles(self):
        sound = crowd_with_whistles(whistles=())
        self.assertEqual(audio.whistles(audio.frame_features(sound, 0.0, 60 * 25)), [])

    def test_offset_shifts_frames(self):
        sound = crowd_with_whistles()
        events = audio.whistles(audio.frame_features(sound, 2.0, 62 * 25))
        self.assertLessEqual(abs(events[0][0] - 12.0 * 25), 2)   # audio starts 2 s into video

    def test_gliding_tone_is_not_a_whistle(self):
        t = np.arange(10 * RATE) / float(RATE)
        sound = 0.2 * np.random.RandomState(0).randn(len(t))
        on = (t >= 4) & (t < 5)
        sweep = 2000 + 3000 * (t[on] - 4)                       # 2 -> 5 kHz in a second
        sound[on] += 0.3 * np.sin(2 * np.pi * np.cumsum(sweep) / RATE)
        self.assertEqual(audio.whistles(audio.frame_features(sound.astype(np.float32), 0.0,
                                                             250)), [])

    def test_decode_mono_reads_an_mp4_audio_track(self):
        import av
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, "clip.mp4")
        container = av.open(path, "w")
        video = container.add_stream("mpeg4", rate=25)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        sound = container.add_stream("aac", rate=RATE)
        tone = (0.3 * np.sin(2 * np.pi * 3000 * np.arange(2 * RATE) / RATE)).astype(np.float32)
        for i in range(50):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
        block = 1024
        for i in range(0, len(tone) - block + 1, block):
            frame = av.AudioFrame.from_ndarray(tone[None, i:i + block], format="fltp",
                                               layout="mono")
            frame.sample_rate = RATE
            frame.pts = i
            for packet in sound.encode(frame):
                container.mux(packet)
        for stream in (video, sound):
            for packet in stream.encode(None):
                container.mux(packet)
        container.close()
        samples, offset = audio.decode_mono(path)
        self.assertGreater(len(samples), 1.5 * RATE)
        self.assertLess(abs(offset), 0.2)
        spectrum = np.abs(np.fft.rfft(samples[RATE // 2:RATE // 2 + 4096]))
        self.assertAlmostEqual(np.fft.rfftfreq(4096, 1.0 / RATE)[np.argmax(spectrum)], 3000,
                               delta=10)


class WhistleDecoderTests(unittest.TestCase):
    def test_whistle_moves_the_stoppage_start_onto_it(self):
        e = np.zeros((300, 3))
        e[:140, 0] = 0.3                         # home has it until the whistle at 140
        e[140:160, 2] = -0.2                     # nothing clear for 0.8 s, mildly "live"
        e[160:, 2] = 1.0                         # then dead-ball clues (players stop, etc.)
        e[:140, 2] = -0.5
        cost = [[0, 10, 4], [10, 0, 4], [4, 4, 0]]
        plain = decoder.decode(e, [25, 25, 25], cost)
        events = [(140, 150)]
        entry, leave = audio.boundary_bonus(events, 300, pre=5, post=25)
        whistled = decoder.decode(e + np.c_[np.zeros((300, 2)), audio.dead_evidence(events, 300)],
                                  [25, 25, 25], cost, entry_bonus=entry, exit_bonus=leave)
        first_dead = lambda labels: int(np.flatnonzero(labels == decoder.D)[0])
        self.assertGreaterEqual(first_dead(plain), 155)
        self.assertLessEqual(abs(first_dead(whistled) - 140), 3)

    def test_whistle_cannot_stop_clearly_live_play(self):
        e = np.zeros((300, 3))
        e[:, 0] = 0.3
        e[:, 2] = -1.0                           # ball seen moving fast the whole time
        events = [(140, 150)]                    # e.g. a crowd whistle
        entry, leave = audio.boundary_bonus(events, 300, pre=5, post=25)
        labels = decoder.decode(e + np.c_[np.zeros((300, 2)), audio.dead_evidence(events, 300)],
                                [25, 25, 25], [[0, 10, 4], [10, 0, 4], [4, 4, 0]],
                                entry_bonus=entry, exit_bonus=leave)
        self.assertTrue(np.all(labels == decoder.H))

    def test_pipeline_switch_without_features(self):
        original = audio.load_features
        audio.load_features = lambda *a, **k: None
        try:
            labels, parts = pipeline.label(synthetic_match(), events=kickoff_events(),
                                           options={"whistle": True})
        finally:
            audio.load_features = original
        self.assertEqual(parts["whistle"], {"available": False})
        self.assertEqual(labels[11185 + 1000], "H")


if __name__ == "__main__":
    unittest.main()
