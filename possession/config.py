"""Every parameter of the pipeline, each with where it came from.

Rule for this file: values are set from the literature, the Laws of the Game or physics, and are
fixed BEFORE any submission is scored on the public window. The public window is a correctness
check (alignment, validity, systematic lag), never a tuning set -- see RULES.md and EVALUATION.md.

Units: metres, seconds, frames at 25 fps. Log-likelihoods are natural log, per frame.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 25

# -- paths --------------------------------------------------------------------------------------
# Derived arrays live here. Ignored by git: they are frame-level derivatives of the gated data.
CACHE_DIR = os.path.join(ROOT, "cache")

# -- raw ATD format -----------------------------------------------------------------------------
# "<frame_id>:<slot>;<slot>;...:<ball>", each group comma-separated floats, NaN when missing.
# Same convention starter/baseline.py parses. The number of fields per group is read from the
# file, not assumed, so a ball group carrying z or a state flag is kept rather than dropped.
FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","

# -- ball track ---------------------------------------------------------------------------------
# Savitzky-Golay, window 7, order 2: Vidal-Codina et al. 2022 (arXiv 2202.00804), sec. 3.
SAVGOL_WINDOW = 7
# Longest gap filled by straight-line interpolation. 0.5 s: a ball between two observations that
# close is, physically, almost always on one rolling/flying path; longer gaps hide touches.
MAX_INTERP_GAP = 12
# Ball counted out of play once beyond a line by this much. Ball radius is 0.11 m (Law 2); the
# rest is allowance for automatic-tracking position error, which is of the order of 0.5 m.
OOB_MARGIN_M = 1.0
# Ball speeds (m/s): below STATIONARY the ball is placed/still; above MOVING it is clearly struck
# or carried. A jogging dribble is ~3 m/s, a placed ball jitters well under 1 m/s.
BALL_STATIONARY_MS = 0.7
BALL_MOVING_MS = 3.0

# -- control evidence -----------------------------------------------------------------------------
# Possession zone radius: 0.5-1.0 m depending on provider in Vidal-Codina et al. 2022. The larger
# value is used because this tracking is automatic and uncorrected.
POSSESSION_ZONE_M = 1.0
# Duel zone: 1 m around the ball, players of both teams inside it -> contested (same source).
DUEL_ZONE_M = 1.0
# Per-frame log-likelihood a frame of clean control adds to that team's state.
CONTROL_LL = 1.0

# -- dead ball ------------------------------------------------------------------------------------
# Window (frames) for smoothing the mean player speed. 2 s: stoppages last several seconds, and a
# short window would follow every sprint.
PLAYER_SPEED_WINDOW = 50
# Weight and clip of the unsupervised (2-component Gaussian mixture, fitted by EM on this match
# without labels) player-speed log-odds. Clipped so it can only tip ambiguous frames.
PLAYER_SPEED_WEIGHT = 0.5
PLAYER_SPEED_CLIP = 2.0
# Ball observed beyond a line: strong per-frame dead evidence.
OOB_LL = 3.0
# Ball observed moving fast inside the pitch: per-frame alive evidence.
MOVING_ALIVE_LL = 1.0
# Ball observed still inside the pitch: mild dead evidence (free kicks, goal kicks, corners).
STATIONARY_DEAD_LL = 0.3
# A tracking gap whose last sighting was this close to a line, lasting at least MIN_GAP frames,
# is evidence the ball left the pitch.
NEAR_LINE_M = 2.0
GAP_OOB_MIN = 25
GAP_OOB_LL = 1.0
# Smart Tagging set piece: dead evidence from the tag start until the ball is first seen moving.
SETPIECE_LL = 1.0

# -- Smart Tagging team prior ---------------------------------------------------------------------
# Phases are not possessions (DATA.md). The weight is set so that a typical ~5 s phase on its
# own adds less than half of one turnover switch cost: it can tip a close call, never make one.
TAG_TEAM_LL = 0.5 * 10.0 / (5 * FPS)
# Codes whose Team tag is the team IN possession, and codes whose Team tag is the team WITHOUT
# the ball (defending phases, and the team that has just lost it).
TAG_ATTACKING = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                 "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX")
TAG_DEFENDING = ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                 "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION")
TAG_HOME_NAMES = ("barcelona",)
TAG_AWAY_NAMES = ("atletic club", "athletic club")

# -- decoder --------------------------------------------------------------------------------------
# Minimum durations and switch costs were first set by principle (1 s minimums; a turnover
# costing 10 = ~0.4 s of clean control). They were then TUNED ON OTHER GAMES WITH REAL ANSWERS:
# open IDSSE Bundesliga matches (Bassek et al. 2025, CC-BY 4.0), tracking degraded to look like
# BAR-ATH's, answers converted to this challenge's "controlled possession" definition
# (EXT_CONTROL_MIN_S). `python -m possession.external tune --joint --control-min 3` on four
# matches, three held out; the finalists were then compared on all seven with a fresh seed under
# both a 2 s and a 3 s conversion, and the best average was kept (possession/NOTES.md).
# Result vs the principle-based values: spellF1 ~0.38 -> ~0.61, macroF1 ~0.80 -> ~0.87, and the
# possession count goes from ~half of the truth to ~the truth. No BAR-ATH answers involved.
# Reading: switching team is fairly cheap, but a possession must last 4 s ("keeps the ball for
# a few seconds", CHALLENGE.md) and leaving a stoppage needs more evidence than entering one.
MIN_POSSESSION = 100
MIN_DEAD = 25
SWITCH_TURNOVER = 2.0
SWITCH_TO_DEAD = 4.0
SWITCH_FROM_DEAD = 8.0
# Boundary snapping: an H<->A switch is moved to the first clean control frame of the new team
# after the old team's last one (CHALLENGE.md: loose balls stay with the last team), searching
# this far either side of the decoded boundary.
SNAP_WINDOW = 4 * FPS

# -- pre-kickoff stretches -----------------------------------------------------------------------
# A graded half can start before its kickoff (FAQ.md: the second half does). Rather than type in
# frame numbers, the kickoff is read from Smart Tagging's automatic "Kick Off (Start)" set-piece
# tag, and every graded frame of that half before it is dead. Only a tag this close to the start
# of the graded half counts, so a mis-tag later in the half cannot blank out live play.
KICKOFF_TYPE = "kick off (start)"
KICKOFF_MAX_DELAY = 5 * 60 * FPS
# Law 8: play starts when the ball is kicked and clearly moves. The tag marks the kickoff moment
# only roughly, so the kickoff is the first frame at or after the tag where the ball is seen
# moving (BALL_MOVING_MS), searched this far; if none, the tag start is used.
KICKOFF_SEARCH = 10 * FPS

# -- improvements, each switched on only after it passes the no-ground-truth gates ----------------
# (python -m possession.masktest and the Smart Tagging agreement in the run diagnostics).

# Restarts. Laws 15-17: when the ball leaves the pitch, the opponents of the team that last
# touched it restart (throw-in, goal kick or corner). The prior is spread over the first
# RESTART_WINDOW frames after the stoppage, and in total it is worth one turnover switch cost:
# enough to pick the restarting team when the post-restart evidence is ambiguous, but not
# enough to override clear control by the other side.
USE_RESTARTS = False
RESTART_WINDOW = 2 * FPS
RESTART_LL = 10.0 / RESTART_WINDOW  # the original turnover cost; kept fixed when that was retuned
# How far back from the moment the ball went out to look for the last team in control.
RESTART_LOOKBACK = 10 * FPS

# Ball-free team evidence. A logistic model maps team shape (who is pushed up, stretched, moving
# forward) to H vs A. It is fitted on this match's own frames where the ball is known, using the
# first-pass labels as pseudo-labels (no ground truth), then applied only where the ball is
# unknown. Per-frame log-likelihood ratios are scaled by 1/FPS because consecutive frames are
# nearly identical: about one independent observation per second.
USE_BALLFREE = False
BALLFREE_WEIGHT = 1.0 / FPS
BALLFREE_CLIP = 3.0
BALLFREE_L2 = 1.0
# Half-width (frames) of the centred difference used for each team's forward push.
BALLFREE_PUSH_HALF = 6

# Possession-zone calibration. A 2-component mixture on log(ball-to-nearest-player distance)
# over this match's in-play frames separates "at a player's feet" from "travelling"; the radius
# is where the two components are equally likely. Clipped to the 0.5-2 m range around the
# literature values (Vidal-Codina et al. 2022) to guard against a degenerate fit.
CALIBRATE_ZONE = False
ZONE_RANGE_M = (0.5, 2.0)

# Long ball gaps as dead-ball evidence. possession.probe on this match (aggregates only, no
# labels): gap lengths are bimodal -- 17 gaps under 1 s, one between 1 and 5 s, 132 of 5 s or
# more (median 10.7 s); the ball is missing on ~41% of graded frames, close to the usual share of
# ball-out-of-play time in elite football; and players move at half the speed (median 1.0 vs
# 2.0 m/s) while it is missing. So the provider largely stops tracking the ball in stoppages.
# The threshold sits in the empty 1-5 s valley of that distribution; the weight equals the
# other gap cue, so very fast player movement (the player-speed cue, up to -1) can still
# keep a gap alive.
USE_GAP_DEAD = False
GAP_DEAD_MIN = 2 * FPS
GAP_DEAD_LL = 1.0

# Cleaner dead-ball evidence. Run diagnostics (aggregates only): 94 of 204 decoded stoppages
# were under 5 s, the ball was visible in most of them, and they were backed only by "ball
# still" or by players slowing down -- both of which also happen in live play (a ball under a
# player's sole, a lull in tempo). Two fixes from the Laws and from probe facts:
#  - a restart needs a stationary, placed ball (Laws 13-17), which takes seconds to set up, so
#    the still cue only counts inside still runs of at least STILL_MIN frames;
#  - the provider tracks the ball mostly while in play (probe), so with the ball seen inside
#    the pitch, slow players alone may not argue for dead (their alive evidence still counts).
# Adopted 2026-11-07 on proxy checks; switched OFF 2026-11-10 when real answers on the IDSSE
# matches showed it costs D recall (stoppages where the ball stays visible). See NOTES.md.
USE_CLEAN_DEAD = False
STILL_MIN = 2 * FPS

# Throw-ins: the ball is held at or just over the touchline, so "inside the pitch" by the 1 m
# out-of-bounds margin. With USE_LINE_AWARE the clean-dead rule (slow players cannot argue for dead
# while the ball is seen inside) only applies when the ball is clearly inside, i.e. more than the
# same OOB_MARGIN_M from every line.
# Only modifies the clean-dead rule, so it has no effect while USE_CLEAN_DEAD is off.
# (It helped together with clean-dead in the first IDSSE search; see NOTES.md.)
USE_LINE_AWARE = False

# Restarting team from Smart Tagging: a SET PIECES tag with a Team names the team taking it.
# Added as the same restart prior (RESTART_LL over RESTART_WINDOW) from the first first-pass alive
# frame at or after the tag start, searched up to TAG_RESTART_SEARCH past the tag end.
USE_TAG_RESTARTS = False
TAG_RESTART_SEARCH = 5 * FPS

# Touch detection (Vidal-Codina et al. 2022: a gain needs the ball's motion to change at a player).
# A touch is a local peak in the change of the ball's velocity vector between TOUCH_HALF frames
# before and after (the Savitzky-Golay window spreads a kick over ~7 frames, so compare outside
# it). A free ball loses well under 1 m/s over that interval to friction and drag, so a change of
# at least BALL_MOVING_MS means something struck it. It is credited to the nearest player within
# TOUCH_RADIUS_M (about twice the 1.1 m typical ball-to-possessor offset in this tracking -- the
# probe's near mode -- for position error plus leg reach), only if the nearest opponent is at
# least TOUCH_MARGIN_M (the duel zone) farther away. Each touch is worth half a turnover switch:
# two touches by the new team (receive + play on: "one or two consecutive actions") justify one.
USE_TOUCHES = False
TOUCH_HALF = 3
TOUCH_DV_MS = BALL_MOVING_MS
TOUCH_RADIUS_M = 2.5
TOUCH_MARGIN_M = DUEL_ZONE_M
TOUCH_LL = 10.0 / 2.0  # half the original turnover cost; kept fixed when that was retuned

# Soft control: instead of all-or-nothing at POSSESSION_ZONE_M, control evidence fades linearly to
# zero at twice that distance, and only counts when the nearest opponent is at least
# TOUCH_MARGIN_M farther than the nearest team-mate (so a passer-by next to an opponent earns
# nothing -- the failure of the plain 2 m zone).
USE_SOFT_CONTROL = True  # chosen by the IDSSE real-answer search, see NOTES.md

# Turnover check (diagnostic only, never an input): Smart Tagging's transition phases mark open-play
# changes of possession. ATTACKING TRANSITION names the team that just won the ball, DEFENSIVE
# TRANSITION the team that just lost it.
TAG_GAIN = ("ATTACKING TRANSITION",)
TAG_LOSS = ("DEFENSIVE TRANSITION",)

# -- video ball (possession/video_ball.py detections -> possession/videotrack.py) -----------------
# Fill ATD ball gaps with a physically consistent path through the per-frame detector candidates.
# Physics: the fastest shots are ~35 m/s, so between two detections k frames apart the ball moves
# at most VIDEO_VMAX * k / FPS, plus VIDEO_POS_TOL for detector/homography position error (the
# benchmark: video vs ATD ball median 0.97 m, p90 ~3 m). A chain may skip up to VIDEO_MAX_SKIP
# frames without a detection (same 0.5 s as MAX_INTERP_GAP), paying VIDEO_MISS per skipped frame.
# Each detection scores log(conf / VIDEO_CONF_REF): above the reference confidence it adds, below
# it subtracts. A new chain not connected to the last ATD sighting pays VIDEO_START, so short,
# isolated or low-confidence chains (heads, boots, line marks) are not kept.
# These values are checked (not tuned to labels) by `python -m possession.videotrack check`,
# which hides ATD-seen stretches and measures how close the fill lands to the hidden ATD ball.
USE_VIDEO_BALL = True  # adopted 2026-11-11, masktest, see NOTES.md
VIDEO_DETECTIONS = os.path.join(CACHE_DIR, "video_ball_yolo11s_1920.npz")
VIDEO_VMAX = 35.0
VIDEO_POS_TOL = 2.0
VIDEO_MAX_SKIP = 12
VIDEO_MISS = 0.1
VIDEO_CONF_REF = 0.25
VIDEO_START = 2.0
VIDEO_OFF_PITCH_M = 3.0

# -- referee whistle (possession/audio.py) --------------------------------------------------------
# Pea and pealess referee whistles have their fundamental around 2.5-4.5 kHz; crowd noise is broad.
# Per video frame, a 64 ms Hann window of mono 16 kHz audio gives the peak-to-median power ratio
# (dB) inside WHISTLE_BAND_HZ ("tonality"). Frames whose robust z-score (median/MAD over the match)
# reaches WHISTLE_Z are whistle-like; runs of them (1-frame holes allowed) lasting WHISTLE_MIN to
# WHISTLE_MAX frames with a steady pitch (peak-frequency std <= WHISTLE_PITCH_STD_HZ) are events.
# A blast is typically 0.2-1 s; long end-of-half blasts run to a few seconds.
AUDIO_FEATURES = os.path.join(CACHE_DIR, "audio_features.npz")
AUDIO_RATE = 16000
AUDIO_WINDOW = 1024
WHISTLE_BAND_HZ = (2000.0, 5000.0)
WHISTLE_Z = 6.0
WHISTLE_MIN = 4
WHISTLE_MAX = 5 * FPS
WHISTLE_PITCH_STD_HZ = 150.0
# A whistle marks a stoppage boundary: play stops on it, or (kickoffs, some free kicks) restarts
# just after it. Within WHISTLE_PRE frames before to WHISTLE_POST after its start, entering or
# leaving D costs nothing extra (the bonus equals SWITCH_TO_DEAD / SWITCH_FROM_DEAD).
# Also, the second after a whistle is dead either way: play has just stopped, or (restart
# whistle) the kick only follows 0.5-2 s later. That second gets WHISTLE_DEAD_LL per frame: 12.5 in
# total, enough to move a boundary onto the whistle, but -1 per frame of fast-moving ball (clearly
# live play) outweighs it, which limits the damage of a crowd whistle mistaken for the referee's.
USE_WHISTLE = False
WHISTLE_PRE = 5
WHISTLE_POST = FPS
WHISTLE_DEAD_LL = 0.5

# -- making other games' tracking look like BAR-ATH (possession/external.py) --------------------
# From `python -m possession.probe gaps` on BAR-ATH (aggregates, our predicted labels, no ground
# truth): 53.6% of stoppages have no ball at all and 73.2% are mostly without it; live play loses
# the ball in 34 gaps over ~61 live minutes, median 5.4 s, p90 9.5 s (lognormal sigma
# ln(9.5/5.4)/1.2816 = 0.44). Ball position error is correlated noise over about a second, its
# size set so the ball-to-nearest-player "at the feet" mode matches BAR-ATH's 1.14 m (probe).
EXT_DEAD_FULL = 0.536
EXT_DEAD_MOSTLY = 0.732
EXT_LIVE_GAPS_PER_MIN = 34 / 61.0
EXT_LIVE_GAP_MEDIAN_S = 5.4
EXT_LIVE_GAP_SIGMA = 0.44
EXT_NOISE_TAU = FPS
EXT_TARGET_NEAR_MODE_M = 1.14
EXT_BALL_NOISE_M = 0.25  # chosen by `python -m possession.external noise`

# Converting other games' answers to THIS challenge's definition before tuning. CHALLENGE.md: a
# change of possession needs *controlled* possession ("a player keeps the ball for a few
# seconds", "more than two consecutive passes"); a touch without control, a deflection or an
# aerial duel without control does not count, and contested/loose phases stay with the last team.
# DFL's per-frame BallPossession flips on such brief touches. So in the IDSSE answers, an H/A spell
# shorter than EXT_CONTROL_MIN_S that directly follows the other team's spell is given back to
# that team. Spells starting from a stoppage are kept (CHALLENGE.md: every restart has a clear
# possessor). "A few seconds" -> 3 s; 2 s is run as a robustness check.
EXT_CONTROL_MIN_S = 3.0
