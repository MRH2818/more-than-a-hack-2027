# Competition analysis — More Than a Hack 2027

Planning date: **November 5, 2026**. Repository snapshot: **`f8f2629`**. This is a planning audit, not an implementation or submission. All competition conclusions below come from the supplied repository; strategic recommendations and unresolved assumptions are explicitly distinguished.

## 1. Decision-critical findings

There is **one fully specified challenge: Phase 1, frame-level football possession**. Teams produce an algorithmically generated CSV covering a complete match. Ranking is automatic, principally by possession-interval `spellF1`; there is **no Phase 1 jury, originality score, presentation score, hardware requirement or LLM requirement**. Consequently, alternative projects in the strategy are competing approaches to this one challenge, not invented official challenge tracks. [README, “The task,” lines 13–29](/Users/danieleitan/Desktop/more-than-a-hack-2027/README.md:13); [OVERVIEW, “Evaluation criteria,” lines 207–221](/Users/danieleitan/Desktop/more-than-a-hack-2027/OVERVIEW.md:207).

The match assets are **not present**: `data/` contains only its README, and `MTAH_DATA_DIR` is unset. The public five-minute labels and four baseline outputs are available. No independent validation match, pretrained checkpoint, GPU entitlement or account access is supplied. Public labels may check pipeline correctness but **must not train, tune or select model versions**. These constraints dominate feasible planning.

## 2. Exhaustive inventory and source map

The recursive inventory contains **34 original non-Git files**. There are no supplied PDFs, Word documents, slide decks, standalone images or videos; all 56 notebook cells were inspected, with no executed cells, embedded outputs or attachments. The gated data bundle is absent. Git internals were not treated as additional competition documents. New planning documents are excluded from this count.

| Original files | Count | Role in the audit |
|---|---:|---|
| [README.md](/Users/danieleitan/Desktop/more-than-a-hack-2027/README.md), [OVERVIEW.md](/Users/danieleitan/Desktop/more-than-a-hack-2027/OVERVIEW.md) | 2 | Scope, calendar, format changes, institutional context |
| [CHALLENGE.md](/Users/danieleitan/Desktop/more-than-a-hack-2027/CHALLENGE.md), [RULES.md](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md), [EVALUATION.md](/Users/danieleitan/Desktop/more-than-a-hack-2027/EVALUATION.md), [SUBMISSION.md](/Users/danieleitan/Desktop/more-than-a-hack-2027/SUBMISSION.md), [FAQ.md](/Users/danieleitan/Desktop/more-than-a-hack-2027/FAQ.md), [DATA.md](/Users/danieleitan/Desktop/more-than-a-hack-2027/DATA.md) | 6 | Binding task, permissions, metrics, hand-in contract, data specifications |
| [LICENSE](/Users/danieleitan/Desktop/more-than-a-hack-2027/LICENSE), [CITATION.cff](/Users/danieleitan/Desktop/more-than-a-hack-2027/CITATION.cff), [.gitignore](/Users/danieleitan/Desktop/more-than-a-hack-2027/.gitignore) | 3 | License scope, citation request, asset/output exclusion patterns |
| [games.py](/Users/danieleitan/Desktop/more-than-a-hack-2027/games.py), [requirements.txt](/Users/danieleitan/Desktop/more-than-a-hack-2027/requirements.txt), [requirements-starter.txt](/Users/danieleitan/Desktop/more-than-a-hack-2027/requirements-starter.txt) | 3 | Constants, asset locations, runtime requirements |
| [possession_metrics.py](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/possession_metrics.py), [run_evaluation.py](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/run_evaluation.py), [validate_submission.py](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/validate_submission.py) | 3 | Executable metric, validation and ranking specification |
| [public GT CSV](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/gt/BAR-ATH_possession_gt_public.csv), [public GT metadata](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/gt/BAR-ATH_possession_gt_public_meta.json) | 2 | Official limited correctness example, not development labels |
| [results README](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/results/README.md), [baseline_raw.csv](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/results/baseline_raw.csv), [baseline_smooth25.csv](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/results/baseline_smooth25.csv), [baseline_smooth50.csv](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/results/baseline_smooth50.csv), [baseline_smooth125.csv](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/results/baseline_smooth125.csv) | 5 | Supplied reference predictions and output conventions |
| [baseline.py](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/baseline.py), [load_atd.py](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/load_atd.py), [load_smart_tagging.py](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/load_smart_tagging.py), [load_homography.py](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/load_homography.py), [utils.py](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/utils.py), [make_dummy_submission.py](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/make_dummy_submission.py) | 6 | Reference algorithms, parsers, projections, plotting and synthetic fixtures |
| [01_smart_tagging.ipynb](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/01_smart_tagging.ipynb), [02_tracking_atd.ipynb](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/02_tracking_atd.ipynb), [03_first_submission.ipynb](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/03_first_submission.ipynb) | 3 | Guided examples; all code cells unexecuted, with no embedded outputs |
| [data/README.md](/Users/danieleitan/Desktop/more-than-a-hack-2027/data/README.md) | 1 | Enrollment/download instructions; confirms expected gated assets |

Authority: the overview defers to the repository’s data, evaluation and rules documents; the **metric implementation prevails over metric prose**. Enrollment terms, absent here in full, govern match-data permissions. [.gitignore](/Users/danieleitan/Desktop/more-than-a-hack-2027/.gitignore:39) contains stale commentary about a README supersession sentence and an empty results directory; comments do not create new rules. [OVERVIEW, lines 9–12](/Users/danieleitan/Desktop/more-than-a-hack-2027/OVERVIEW.md:9); [EVALUATION, lines 146–153](/Users/danieleitan/Desktop/more-than-a-hack-2027/EVALUATION.md:146).

## 3. Challenge, calendar and eligibility

The organizers are Metrica Sports, FC Barcelona/Barça Innovation Hub and Mobile World Capital Barcelona. The provided match is Barcelona versus Athletic Club, LaLiga 2024–25 matchday 2. Barcelona maps to `H`, Athletic to `A`. The canonical tactical video has **199,319 frames at 25 fps, 1920×1080**. Every frame needs a row; 158,857 frames are graded. [DATA, opening table](/Users/danieleitan/Desktop/more-than-a-hack-2027/DATA.md:3).

| Official event | Date |
|---|---|
| Registration opens | October 1, 2026 |
| Data release | October 20, 2026 |
| Submission deadline | November 17, 2026 |
| Results announced | Before November 26, 2026 |

These **2026 dates are explicit despite the 2027 event title**. No deadline hour/timezone or fixed 24/48-hour building period is specified. November 5 is 12 calendar days before the dated deadline. A proposed November 15 upload is a team buffer, not an official cutoff. [README, “Timeline”](/Users/danieleitan/Desktop/more-than-a-hack-2027/README.md:164).

Teams “must be composed of 3 to 5 members,” followed by a parenthetical excluding smaller teams from the “Innovation Challenge and the Final Pitch.” Treat 3–5 as the conservative requirement. Those names do **not** supply a second challenge specification: Phase 2 objectives, dates, qualification, rubric and pitch rules remain unknown. [RULES, “Eligibility and teams”](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md:93).

### Possession semantics

| Label | Meaning |
|---|---|
| `H` / `A` | Home/away controlled possession with ball alive |
| `D` | Dead ball; no team ownership required |
| `X` | Prediction abstention, or an ungraded frame in GT |

During loose/contested live play, retain the last controlling team. A simple touch, deflection, or won aerial duel without subsequent control does not establish a turnover. The examples “a few seconds” or “more than two passes” are suggested proxies, **not mandated control thresholds**. Ground truth contains only H/A/D on graded frames and derives mechanically from corrected tracking. A possession is a maximal H/A run: `H D H` contains two possessions. [CHALLENGE, “Possession rules,” lines 49–103](/Users/danieleitan/Desktop/more-than-a-hack-2027/CHALLENGE.md:49).

## 4. Mandatory, optional and assumed requirements

| Status | Requirement or recommendation | Authority |
|---|---|---|
| Mandatory | Algorithm produces all labels; no manual labels, corrections or transition annotations on this match for any purpose | [RULES 41–71](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md:41) |
| Mandatory | No private GT, its derivatives, or human-produced event/tracking/possession data about this match from any source | [RULES 73–84](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md:73) |
| Mandatory | Public five minutes only check alignment, formatting and semantic correctness; no training, tuning or version selection | [EVALUATION 139–142](/Users/danieleitan/Desktop/more-than-a-hack-2027/EVALUATION.md:139) |
| Mandatory | Organizers can access, inspect and run the method; technical document explains dependencies and approach | [RULES 3–21](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md:3) |
| Mandatory | Public repository reproduces exactly the submitted file, with pinned dependencies and instructions | [SUBMISSION 51–60](/Users/danieleitan/Desktop/more-than-a-hack-2027/SUBMISSION.md:51) |
| Mandatory | Do not redistribute match assets or reconstructive derivatives, including clipped video or frame-level tracking exports | [RULES 100–114](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md:100) |
| Mandatory | No restricted competition material or extracts uploaded/disclosed to third-party hosted/cloud LLMs | [FAQ 109–112](/Users/danieleitan/Desktop/more-than-a-hack-2027/FAQ.md:109) |
| Permitted, optional | Classical methods, accessible pretrained models/APIs, other-match labels, algorithmic tracking improvement | [RULES 3–35, 61–68](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md:3) |
| Permitted, optional | Current-match unsupervised/self-supervised learning and algorithmic pseudo-labels with no human-selected labels | [RULES 65–71](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md:65) |
| Recommended | Validate exact final upload; understand and beat published baseline | [SUBMISSION 33–40](/Users/danieleitan/Desktop/more-than-a-hack-2027/SUBMISSION.md:33); [EVALUATION 116](/Users/danieleitan/Desktop/more-than-a-hack-2027/EVALUATION.md:116) |
| Strategic assumption | CPU-first local offline system, early upload buffer, independent validation matches, internal demo | Team planning choices; neither supplied resources nor scoring requirements |

Watching video and inspecting data for generic features/debugging is allowed; converting observations into frame-specific constants, validation labels or corrections is forbidden. Publicly supplied half boundaries are explicitly usable metadata, a different case from newly hand-authored overrides. [RULES 65–71](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md:65); [DATA 249–263](/Users/danieleitan/Desktop/more-than-a-hack-2027/DATA.md:249).

MIT covers repository code, documentation and public GT; it does not grant match-data rights or override competition restrictions on public-label tuning. Preserve copyright/license notices when copying substantial MIT material. Citation is requested, without a stated scoring penalty. [LICENSE](/Users/danieleitan/Desktop/more-than-a-hack-2027/LICENSE:1); [CITATION.cff](/Users/danieleitan/Desktop/more-than-a-hack-2027/CITATION.cff:3).

## 5. Exact scoring mechanics and baseline comparison

For graded H/A/D frames, `macroF1 = (F1_H + F1_A + F1_D)/3`, where `F1_c = 2TP_c/(2TP_c+FP_c+FN_c)` and a zero denominator produces zero. Predicted X contributes a miss. Frame accuracy is diagnostic, not the ranking objective. [possession_metrics.py, `per_class_prf` and `macro_f1`](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/possession_metrics.py:159).

Temporal scoring first finds contiguous **GT non-X runs**; on full GT these are halves, on public GT just its labeled window. Within each run, predicted X forward-fills then backfills at the beginning. A completely abstained run remains X because there is no value to carry. Runs never bridge halftime. Therefore “X costs nothing” is false in general, particularly across a turnover. [possession_metrics.py, `valid_runs`/`fill_abstentions`](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/possession_metrics.py:259).

For same-team, overlapping half-open possession intervals `G=[g0,g1)` and `P=[p0,p1)`:

```text
IoU(G,P) = (min(g1,p1) - max(g0,p0)) / (max(g1,p1) - min(g0,p0))
F1(q)    = 2 TP(q) / (2 TP(q) + FP(q) + FN(q))
spellF1  = mean(F1(q), q in {0.3,0.4,0.5,0.6,0.7})
```

At each threshold, candidate pairs are sorted by descending IoU, then GT/prediction index; a **greedy one-to-one** match consumes each spell once. This is not Hungarian/global optimal matching, not mAP, and not a weighted sum with macroF1. D spells do not themselves count as possessions, but wrong D boundaries split/merge live possessions. [possession_metrics.py, `possession_spells` through `spell_scores`](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/possession_metrics.py:464).

Actual ranking order in [`run_evaluation.rank_key`](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/run_evaluation.py:165) is:

1. Higher `spell_f1_mean`.
2. Higher `macro_f1`.
3. Higher transition F1 at the largest configured tolerance (**25 frames** by default; also reports 12).
4. Lower mean absolute matched-transition latency; unavailable latency sorts last.
5. Submission name ascending.

Transition matching is one-to-one and by destination label by default; it includes live/dead boundaries. These additional tie-breaks are executable-specification details, not new subjective criteria. [metric transition methods](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/possession_metrics.py:325); [runner defaults](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/run_evaluation.py:502).

| Published baseline | Full-match macroF1 | Full-match spellF1 | Public spellF1 |
|---|---:|---:|---:|
| raw | 0.6532 | 0.0486 | 0.0378 |
| smooth25 | 0.6752 | 0.1996 | 0.2868 |
| smooth50 | 0.6724 | **0.2433** | 0.3256 |
| smooth125 | 0.6356 | 0.2242 | **0.4242** |

These are **published reference results**, not measurements of a proposed system. Public ordering reverses the meaningful full-match comparison, demonstrating why it cannot select a model. The published “floor” is a target, not an explicit eligibility threshold. [EVALUATION, baseline tables](/Users/danieleitan/Desktop/more-than-a-hack-2027/EVALUATION.md:98); [README, public board](/Users/danieleitan/Desktop/more-than-a-hack-2027/README.md:64).

The implementation is more specific than “majority smoothing”: nearest visible player yields H/A, otherwise D; smoothing counts **H and A only**, outputs D only when neither is present anywhere in the window, and breaks H/A ties toward H. Window numbers are **half-widths**: smooth50 means ±50 frames, 101 total (4.04 seconds), not a 50-frame total window. It lacks genuine dead-ball detection and possession-state reasoning. [baseline.py, `VARIANTS`, `nearest_player_labels`, `smooth`](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/baseline.py:61).

The documented noise bands (~0.01 macroF1, ~0.04 spellF1 on the full match) are context, not a guarantee or permission to compare variants on public GT. Use other-match labels or synthetic tests for development; do not promise a hidden-test score.

## 6. Data, alignment and engineering constraints

| Asset | Available in competition bundle | Limitation |
|---|---|---|
| Tactical video/ATD | 199,319 frames/records; 46 slots | Slots are tracklets, not players; gaps, duplication and identity changes |
| Tactical ball | Visible on 59.3% graded frames, 56.9% overall | Missingness clusters in difficult play; absence is not dead-ball evidence by itself |
| Tactical tagging | 638 automatic phase/set-piece intervals | False positives/negatives; phase intervals are not possession intervals |
| Tactical homography | 199,319 rows, 100% valid flags | Validity and reprojection quality are separate |
| Broadcast video/ATD/homography | 164,488 records; 51 slots; 65.8% valid homographies | Independent timeline, cuts/replays; no alignment mapping supplied |

All bundle entries are **expected but locally missing**. [DATA, asset table and retrieval](/Users/danieleitan/Desktop/more-than-a-hack-2027/DATA.md:8); [data README](/Users/danieleitan/Desktop/more-than-a-hack-2027/data/README.md:1).

The scorer/reference baseline need Python **3.8+**, NumPy and pandas; the notebooks additionally use kloppy and matplotlib, with Python **3.9+ recommended**. These are starter compatibility requirements, not a restriction on the submission language. No hardware purchase, GPU or external API is required by the challenge. [README, “Requirements”](/Users/danieleitan/Desktop/more-than-a-hack-2027/README.md:152).

Coordinates are normalized 0–1; convert both axes before metric distances. Metadata says 105×68 m, while `games.py` uses 105.3×68 m; choose and document one convention. Ball/player NaNs remain missing observations. Homography maps normalized pitch to pixels; it does not establish an airborne ball’s true ground position. [DATA, ATD and homography](/Users/danieleitan/Desktop/more-than-a-hack-2027/DATA.md:69).

Tagging has uppercase codes, title-case `Team`/`Half` fields, away spelling `Atletic Club`, and 36 `N/A` team tags including every throw-in. Defensive tags name the defending team. Set-piece windows provide restart evidence, **not exact whistle or kick frames**: the loader describes approximately second-level precision and containment around kickoffs. [DATA, Smart Tagging](/Users/danieleitan/Desktop/more-than-a-hack-2027/DATA.md:123); [load_smart_tagging.py, lines 37–53](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/load_smart_tagging.py:37).

| Timeline object | Correct conversion |
|---|---|
| Raw supplied ATD `frame_id` | `video_frame = frame_id - 1` |
| Tagging seconds | `round(seconds * 25)`; intervals half-open |
| Homography frame | Already zero-based tactical frame |
| `games.py` large `IdFrame` counter | Separate corrected-tracking/GT derivation counter; **never apply its offsets to raw ATD** |

The `games.py` opening docstring is misleading without context; [`load_atd.py`, lines 24–33](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/load_atd.py:24) explicitly resolves it. A correct-looking CSV can still contain a catastrophically misaligned inference timeline.

Graded inclusive ranges are **11185–90651** and **110808–190197**; 40,462 other frames are ignored. Public GT labels only **11185–18684**: 7,500 frames, 17 possessions; remaining rows X. [DATA, frame alignment/public GT](/Users/danieleitan/Desktop/more-than-a-hack-2027/DATA.md:238).

FAQ puts second kickoff at **113742**, after 2,934 graded dead-ball frames; tagging loader agrees. `games.py` playback anchors calculate **11186 / 113741**, whereas documented first kickoff/start is 11185. Do not shift all data to resolve these one-frame discrepancies or create private frame overrides. Use published grading ranges, raw ATD indexing, and clarify kickoff-anchor semantics. [FAQ 33–41](/Users/danieleitan/Desktop/more-than-a-hack-2027/FAQ.md:33); [games.py, anchors](/Users/danieleitan/Desktop/more-than-a-hack-2027/games.py:56).

## 7. Submission checklist and unresolved questions

The file contract is UTF-8 CSV with exactly `frame,label`, 199,319 data rows, integer frames 0…199318 in order, nonblank H/A/D/X labels, and no extra index. Case/whitespace normalization is tolerated. The parser happens to accept additional columns, but the **documented stricter contract should be followed**. [SUBMISSION 3–40](/Users/danieleitan/Desktop/more-than-a-hack-2027/SUBMISSION.md:3); [metric CSV validation](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/possession_metrics.py:84).

Submit through the emailed Google Form: registered team name, team-named CSV, public repository URL, and a short technical PDF/Markdown document. Google account required; one editable response/team; last pre-deadline response scored. No live leaderboard or feedback exists. Explain method, assets used/dropped, every labeled training dataset and annotator/quantity, possession rules, limitations and reproduction. Organizers may regenerate the winner’s exact file. [SUBMISSION 45–101](/Users/danieleitan/Desktop/more-than-a-hack-2027/SUBMISSION.md:45); [OVERVIEW 216–217](/Users/danieleitan/Desktop/more-than-a-hack-2027/OVERVIEW.md:216).

Resolve these without blocking document planning:

| Question | Interim planning treatment |
|---|---|
| Exact submission cutoff/timezone and form link? | Seek registration email; freeze by November 14 and upload by November 15 as a team buffer |
| Team eligibility wording; any Phase 2 invitation/rules? | Plan for 3–5 people; make no Phase 2 claims |
| Complete enrollment terms; public demo/derived visualization rights? | Keep licensed diagnostic assets local; no public clips/tracking exports |
| Bundle access and integrity? | First development gate; do not pretend data inspection or model validation happened |
| Other-match labeled data accessible to organizers? | Verify license, schema, definition compatibility and availability before dependence |
| Kickoff anchor discrepancies and interval endpoint semantics? | Follow raw frame indexing and official half ranges; request clarification |
| Markdown permitted but filename example requests `.pdf`? | Final technical PDF is the conservative hand-in choice |
| Runtime/memory limits, target execution environment, package/model retrieval? | None specified; set transparent team targets and reproducible CPU fallback |

No rule establishes points for novelty, engineering complexity, visual polish, hardware or AI. The strongest strategy earns a better possession file while satisfying exact reproducibility and data-use constraints; demonstrations support understanding and debugging rather than substitute for performance.
