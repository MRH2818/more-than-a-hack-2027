"""The one command: provided data in, submission CSV out.

    python -m possession.run                          # writes evaluation/results/possession_v1.csv
    python -m possession.run --out path/to/team.csv
    python -m possession.run --use restarts ballfree   # exactly these optional parts on

Deterministic: no randomness anywhere in the pipeline. Prints aggregate diagnostics only.

With the video ball on (config.USE_VIDEO_BALL), the first run downloads the pinned detector
weights and runs ball detection over every graded frame of the tactical video (about 2-3 hours
on an Apple M3 Pro GPU; resumable). Later runs reuse cache/ and take a minute.
"""

import argparse
import os

from . import config, diagnostics, io_atd, pipeline, tagging

DEFAULT_OUT = os.path.join(config.ROOT, "evaluation", "results", "possession_v1.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--use", nargs="*", choices=pipeline.OPTIONS, default=None,
                        help="optional parts to switch on (others off); omit for config defaults")
    args = parser.parse_args()

    options = None if args.use is None else {key: key in args.use for key in pipeline.OPTIONS}
    opts = pipeline.resolve_options(options)
    if opts["videoball"] or opts["videofull"]:
        from . import video_ball
        video_ball.ensure_weights()
        video_ball.detect(config.VIDEO_DETECTIONS)    # resumes; "nothing to do" when complete
    arrays = io_atd.load(rebuild=args.rebuild_cache)
    events = tagging.load_events()
    labels, parts = pipeline.label(arrays, events=events, options=options)
    path = pipeline.write_csv(args.out, labels)
    diagnostics.print_summary(diagnostics.summarise(labels, parts, events))
    print("\nwrote {}".format(os.path.relpath(path, config.ROOT)))
    print("next: python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, config.ROOT)))
