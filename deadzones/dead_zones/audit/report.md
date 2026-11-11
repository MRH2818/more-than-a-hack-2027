# Remaining stop/restart audit

**No frames are certainly dead from this XML alone.** A phase's end is not an observed stop, and a restart tag is a clip rather than the restart instant. Trimming a guard does not change that. No probability or safe subrange can be established from these tags alone.

## Gap interiors

15 non-goal candidate gaps contain 5305 frames (212.20s). 0 are confirmed dead by the optional reference; 5305 need additional evidence. 6 gaps overlap another restart clip.

| Pair | Restart | Frames [start, end) | Seconds | Certain dead | Needs evidence | Other restart clips |
|---|---|---|---|---|---|---|
| pair-005 | throw-in | 19158–19510 | 14.08 | 0 | 352 | 454 |
| pair-006 | throw-in | 21236–21551 | 12.60 | 0 | 315 | — |
| pair-007 | throw-in | 27463–27769 | 12.24 | 0 | 306 | — |
| pair-008 | goal kick | 50872–50975 | 4.12 | 0 | 103 | — |
| pair-009 | goal kick | 84710–84822 | 4.48 | 0 | 112 | — |
| pair-010 | throw-in | 85230–85548 | 12.72 | 0 | 318 | 452 |
| pair-011 | throw-in | 120471–120599 | 5.12 | 0 | 128 | — |
| pair-012 | throw-in | 123553–123579 | 1.04 | 0 | 26 | — |
| pair-013 | throw-in | 126972–127227 | 10.20 | 0 | 255 | — |
| pair-014 | throw-in | 165546–165834 | 11.52 | 0 | 288 | 430 |
| pair-015 | throw-in | 168770–169699 | 37.16 | 0 | 929 | — |
| pair-016 | goal kick | 180976–182388 | 56.48 | 0 | 1412 | — |
| pair-017 | throw-in | 182769–182837 | 2.72 | 0 | 68 | 456 |
| pair-018 | throw-in | 184944–185416 | 18.88 | 0 | 472 | 447, 459 |
| pair-019 | throw-in | 188186–188407 | 8.84 | 0 | 221 | 445 |

## Full review scope

Includes all 53 ordinary restart windows, the inferred gaps, and their trimmed margins. Overlapping windows count once: 47804 reviewed frames; 667 reference-dead, 485 reference-live, 46652 unresolved.

Reference: `/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/gt/BAR-ATH_possession_gt_public.csv`. Only non-X reference labels establish certainty here. This is a reference audit, not a prediction or measured model accuracy. Reference labels never enter the inferred D mask or possession overlay.

Confirmed-dead ranges within this scope (exclusive ends):
- Frames **17533–17838** (12.20s).
- Frames **18027–18389** (14.48s).

## Evidence needed to mark more frames

- Gap interiors: establish the actual ball-out/whistle frame and uninterrupted dead play through the candidate; a phase ending or missing ball track is insufficient.
- Restart windows: locate the actual restart contact; the window may contain both dead and live frames. A play tag overlapping the window adds ambiguity, not proof of live play.
- Other restart clips overlapping a gap: resolve their ordering and whether play resumed between the two events. The clipped windows may merely overlap because of padding.

For an independently verified, uninterrupted stop/restart pair with bounded times, only frames from the **latest possible stop** through the **earliest possible restart** (exclusive) are guaranteed dead. Frames outside that intersection need boundary evidence. The XML supplies no validated stop bounds for these gaps.

`frames.csv` lists every reviewed frame with status, source IDs, and missing evidence. `segments.csv` compresses identical adjacent rows. `gaps.csv` summarizes the original non-goal candidates. `reference_dead_intervals.csv` contains reference-confirmed dead ranges only. Frames absent from this audit are outside its scope, not necessarily live.
