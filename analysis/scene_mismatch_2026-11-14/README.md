# Initial-scene discrepancies: provenance, corrections, and sensitivity

Audit date: November 14, 2026. **Original model experiments were affected.** This was not introduced by the public export. Historical inputs and scientific outputs remain byte-for-byte unchanged; the new images and index are prospective corrections.

## Root cause and authority

The two reported cached screenshots were recovered from `environment_set_showcase_v4` and match both the original cloud PNGs and the image digests saved in actual model requests. The July 25 pipeline copied the VQA staging images into `static_screenshots/base` and paired them, by family/environment filename, with worlds from a separately maintained simulation-runtime directory. The historical asset builder tested file existence and hashes but never tested geometry; it unconditionally emitted `inconsistent_assets: []`. The public exporter later copied those same files faithfully.

| Layout | Historical image | Executed simulator world | Frozen cells |
| --- | --- | --- | --- |
| BackUp_8 | Green center (96.5, 570.5) | Green center (505, 307) | 013, 014 |
| Balance_8 | Green center (401.5, 277.5) | Green center (548, 271) | 031, 032 |
| Falling_1 (new integrity finding) | Blue bounds x=209–361, y=176–253 | Blue rendered bounds x=216–354, y=183–253 | 063, 064 |

Coordinates are bottom-left pixels. The original red/green-center check missed the Falling_1 compound-body discrepancy. A native full-image comparison found exactly these three layouts; the other 63 historical RGB images are already identical to native renders.

For historical interpretation, the inconsistent image/world pair is the authoritative record: the agent saw one geometry and its action was executed/scored in another. No retrospective pixel replacement or re-scoring can recover what the model would have done with the right image. For prospective corrections, the executed worlds are retained because their geometry is independently supported by run-asset hashes, archived JSON initial-scene prompts, and the task-generation/evaluation code—not merely because they are machine-readable.

The exact earlier world revision that produced the old showcase images has not been recovered. This limits dating the original divergence, but does not limit the evidence that the inconsistency was present before the affected model runs and faithfully propagated into the release.

Evidence: [source chain](provenance/source_chain.json), [historical builder](provenance/historical_build_full_asset_index.py), [representative actual requests](provenance/representative_model_request_evidence.json), [full archived-unit inventory](request_affected_archived_units.csv). No private participant records were read.

## Affected experiments

- Requested pair: 27 distinct goal tasks per main model: BackUp_8 downward/upward = 5/1, Balance_8 downward/upward = 12/9. These comprise 23 noncanonical goals plus four canonical goals.
- Additional Falling_1 finding: 32 goal tasks (18 downward, 14 upward), including two canonical goals. All three layouts therefore affect 59 tasks: 53 noncanonical plus six canonical.
- Confirmed in actual requests for GPT-5, Gemini 3.1 Pro, Qwen 3.6 Plus, Gemini Robotics ER2, and the separately archived older Robotics-ER control. Earlier snapshot rows are repeated historical coverage, not additional participants or independent task replications.
- Visual initial observations affect `full`, `neither`, `frames_only`, and `status_only` wherever those arms were collected, plus their visual novel-action probes. The static initial image is resent on later visual requests; receiving correct simulator rollout frames does not undo that inconsistency.
- `trace_status` has an independent initial action and world-derived JSON geometry, with no image supplied. All 521 recovered JSON initial-scene request geometries checked across affected public snapshots agree with the world files; none of their JSON-arm calls contains images. For fair sensitivity comparisons, the same layout exclusions are applied to both visual and JSON arms. JSON-versus-visual contrasts remain representation controls, not shared-first-action causal feedback contrasts.
- The original archive catalog also matches these screenshots to historical full/compact prompt-control assets and coordinate-VQA staging. The [historical prompt asset ledger](provenance/historical_prompt_asset_pairs.json) records the matching full/compact asset pairs. Their dated records are not silently relabeled as corrected. This audit does not rerun or alter VQA scoring, coordinate perturbations, or historical prompt scoring.

Exact goal IDs, archived source units, request counts, and source shard hashes: [task manifests](request_affected_task_manifests.json), [snapshot/model summary](request_summary.json), [request verification](request_verification.json), and the population files in each sensitivity directory.

## Requested sensitivity: BackUp_8 and Balance_8 excluded

The original 1,641 main tasks become **1,614 tasks / 64 layouts**; the 1,651 broad feedback population becomes **1,624 tasks**. The 118-task transfer subset becomes **114 tasks / 63 layouts / 51 direction pairs**. The canonical 132-task panel becomes 128 tasks.

| Model | Full-feedback solved / tasks | Solve rate %, 95% CI |
| --- | --- | --- |
| GPT-5 | 666/1614 | 41.26 [37.88, 44.58] |
| Gemini 3.1 Pro | 745/1614 | 46.16 [43.06, 49.28] |
| Qwen 3.6 Plus | 660/1614 | 40.89 [37.82, 43.91] |

| Model | Later success: full − retry (n) | Next chosen PQ: full − retry (n) | Early novel PQ: full − retry (n) |
| --- | --- | --- | --- |
| GPT-5 | 0.0607 [0.0332, 0.0878] (n=1318) | 0.0706 [0.0467, 0.0981] (n=1318) | 0.1688 [0.1024, 0.2375] (n=58) |
| Gemini 3.1 Pro | 0.0412 [0.0206, 0.0636] (n=1166) | 0.0088 [-0.0069, 0.0248] (n=1166) | 0.0627 [-0.0152, 0.1429] (n=50) |
| Qwen 3.6 Plus | 0.0178 [-0.0008, 0.0359] (n=1293) | 0.0452 [0.0274, 0.0648] (n=1293) | 0.1002 [0.0309, 0.1718] (n=62) |

All feedback differences above condition on the same failed first action. Novel prediction uses the common one-rollout checkpoint. Success differences are probability units; PQ differences use the unchanged sigma=150 score.

| Model | Figure 4A: adjusted chosen PQ difference | Figure 4B: richer partial Spearman | Figure 4C: adjusted early novel PQ difference |
| --- | --- | --- | --- |
| GPT-5 | 0.0414 [0.0019, 0.0804] | -0.1246 [-0.3593, 0.1044] | -0.0278 [-0.1425, 0.0905] |
| Gemini 3.1 Pro | 0.1007 [0.0761, 0.1276] | -0.0056 [-0.2965, 0.2990] | -0.1275 [-0.2965, 0.0397] |
| Qwen 3.6 Plus | 0.0471 [0.0202, 0.0714] | 0.0825 [-0.1827, 0.3500] | -0.0667 [-0.2598, 0.1305] |

Figure 4A includes 9,255 / 8,528 / 9,197 attempts for GPT / Gemini / Qwen, respectively, with 370 / 297 / 339 informative recovering episodes. Exact-goal and categorical-attempt controls and equal-episode weights are unchanged. Figure 4B retains layout/direction, initial chosen/novel predictions, and rollout-count controls; its simpler layout/direction-only result is also supplied. Figure 4C retains baseline novel PQ and layout/direction controls.

## Expanded integrity sensitivity: all three layouts excluded

The main population is **1,582 tasks / 63 layouts**; broad controls use **1,592 tasks**. Matched transfer uses **112 tasks / 62 layouts / 50 direction pairs**; the canonical panel uses 126 tasks.

| Model | Full-feedback solved / tasks | Solve rate %, 95% CI |
| --- | --- | --- |
| GPT-5 | 654/1582 | 41.34 [38.00, 44.67] |
| Gemini 3.1 Pro | 730/1582 | 46.14 [43.07, 49.33] |
| Qwen 3.6 Plus | 652/1582 | 41.21 [38.17, 44.23] |

| Model | Later success: full − retry (n) | Next chosen PQ: full − retry (n) | Early novel PQ: full − retry (n) |
| --- | --- | --- | --- |
| GPT-5 | 0.0574 [0.0301, 0.0846] (n=1289) | 0.0703 [0.0462, 0.0987] (n=1289) | 0.1688 [0.1024, 0.2375] (n=58) |
| Gemini 3.1 Pro | 0.0412 [0.0201, 0.0642] (n=1140) | 0.0089 [-0.0072, 0.0253] (n=1140) | 0.0627 [-0.0152, 0.1429] (n=50) |
| Qwen 3.6 Plus | 0.0213 [0.0037, 0.0386] (n=1266) | 0.0439 [0.0259, 0.0637] (n=1266) | 0.1002 [0.0309, 0.1718] (n=62) |

| Model | Figure 4A | Figure 4B, richer | Figure 4C |
| --- | --- | --- | --- |
| GPT-5 | 0.0449 [0.0055, 0.0828] | -0.1312 [-0.3686, 0.0978] | -0.0269 [-0.1370, 0.0849] |
| Gemini 3.1 Pro | 0.1027 [0.0772, 0.1295] | -0.0019 [-0.2870, 0.3098] | -0.1274 [-0.2978, 0.0443] |
| Qwen 3.6 Plus | 0.0484 [0.0218, 0.0725] | 0.0872 [-0.2007, 0.3583] | -0.0634 [-0.2478, 0.1392] |

Figure 4A now includes 9,051 / 8,349 / 8,989 attempts. All Figure 4B/C intervals still span zero. The positive chosen-action associations remain modest.

Robotics ER2: removing the requested two layouts gives **96/128 = 75.0%**, with a 95% layout-bootstrap interval [67.19%, 82.81%]; the historical 125-probe subset becomes 121. Removing all three gives **94/126 = 74.60%** [66.67%, 82.54%] and 119 probes. Its adjusted Figure 4A-style chosen-PQ association is supplied separately in the JSON/CSV outputs; no human comparison is added.

## Implications for reporting

1. Disclose the initial-observation/world inconsistency. It can confound comparisons involving visual versus JSON representation, action/prediction correspondence, and learning from visual feedback. It is not a mere packaging typo.
2. For a clean input-validity primary analysis, use the all-three-layout exclusion (1,582 main tasks; 112 matched transfer tasks) and retain the original and requested two-layout estimates as transparent sensitivity records. Alternatively, retain original main estimates only with an explicit limitation and these sensitivity results. No manuscript was edited here.
3. The qualitative Figure 4 conclusion survives: successful action has a modest positive association with prediction of that action; associations involving novel actions remain uncertain after adjustment. These exploratory intervals do not establish absence of coupling.
4. Feedback conclusions are mostly stable, but Qwen’s subsequent-success benefit becomes 0.0213 [0.0037, 0.0386] with all three layouts removed, compared with 0.0178 [−0.0008, 0.0359] in the requested two-layout sensitivity. This is a post hoc sensitivity, not a new preregistered significance claim.
5. Historical full/compact and other earlier controls retain their original populations and known protocol limitations. Do not describe them as run with the corrected assets or as independently cleaned by this release.

## Corrections and preservation

New files are `132_base_environments/corrections/2026-11-14/{BackUp_8,Balance_8,Falling_1}/initial_scene.png` and `task_configs/asset_index_132_scene_corrected_2026-11-14.json`. Six index image references change in this new index. All world and tool references/hashes are retained. The original `cells/` files, `index.csv`, `index.jsonl`, `metadata.json`, original task manifests, model shards, VQA records, and figures remain unchanged.

The [corrected-asset manifest](corrected_asset_manifest.json) contains original and new PNG hashes, native RGB hashes, renderer code hashes and dependency versions. [Historical full-image check](all66_historical_geometry.json) detects exactly three discrepant layouts. [Corrected full-image check](all66_corrected_geometry.json) matches all 66 RGB arrays exactly, including compound blue-body geometry. Native rendering uses Python 3.12, Pygame 2.6.1, and Pymunk 6.11.1; no physics steps, model calls, or synthetic participant data are involved.

## Numerical methods and reproducibility

All analyses use model seed 2026 and the frozen November 14 populations, filtered by layout before fitting. Whole-layout resampling retains all task goals and both directions. Solve-rate intervals use 10,000 draws and the original model-specific seeds; paired feedback intervals use 100,000 draws, seed 20261114. Figure 4 estimators use the unchanged frozen functions and 2,000 draws, seeds 6112702 (A/C) and 6112703 (B), with ranks and nuisance adjustments refitted for partial Spearman. The exact output CSVs and JSONs retain full precision, counts, and adjustment details.

The two-layout analysis was run both from the previously verified reconstruction and afresh from 57 public model shards; every estimate, interval, population, and CSV byte matched. The expanded analysis uses that verified reconstruction with only the additional layout filter.

With `requirements-analysis.txt` installed, reproduce model-only sensitivity:

```bash
python scripts/evaluation/analyze_scene_exclusions.py --output-dir outputs/scenes_two
python scripts/evaluation/analyze_scene_exclusions.py --exclude-layouts BackUp_8 Balance_8 Falling_1 --output-dir outputs/scenes_three
python scripts/release/audit_scene_request_provenance.py --output-dir outputs/scene_requests
```

With simulator dependencies installed, verify:

```bash
python scripts/release/build_task_configs.py --check
python scripts/release/attach_initial_scenes.py --check
python scripts/release/build_scene_corrections.py --check
python scripts/release/verify_scene_geometry.py
python -m unittest discover -s scripts/evaluation -p "test_*.py" -q
python -m compileall -q scripts simulator
shasum -a 256 -c analysis/scene_mismatch_2026-11-14/SHA256SUMS
```

The original-index geometry check intentionally fails on the three known historical discrepancies; it is evidence of the issue, not a failed correction. All prospective checks, the four repository tests, and compilation passed.

## Licensing

MIT code/simulator publication is a separate maintainer-authorized update. See [LICENSE](../../LICENSE), [NOTICE.md](../../NOTICE.md), and [LICENSE_STATUS.md](../../LICENSE_STATUS.md). The upstream authorization correspondence was not independently located; the license is based on the maintainer’s explicit MIT direction and stated permission. Asset-specific upstream scope remains documented separately. No blanket third-party relicensing is asserted.
