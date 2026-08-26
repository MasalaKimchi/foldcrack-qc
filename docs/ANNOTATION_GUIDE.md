# Fold and crack/tear annotation guide

This guide creates a reproducible reference standard for H&E, COMET, and CosMx
image QC. It is intentionally operational: labels describe an observable physical
or acquisition condition and the area it affects, not a guess that “looks bad.”

Reviewers must complete modality-specific training and a calibration round before
annotating study data. The annotation tool, coordinate system, image channels,
windowing, scale, and software version are fixed in the study manifest.

## 1. Decide what “crack” means before labeling

“Crack” is not a sufficiently precise ground-truth class. At least four visually
similar mechanisms can require different actions:

| Subtype | Physical meaning | Typical action |
|---|---|---|
| `tissue_tear` | Split, fissure, shatter, or missing corridor in the tissue section | Recut/restain, review, or mask affected tissue |
| `glass_crack` | Crack/crazing in coverslip or slide glass, including optical distortion across tissue | Remount/rescan/replace slide as workflow dictates |
| `knife_line` | Microtomy score, chatter, repeated parallel line, or blade damage | Recut or review; do not call a tissue tear |
| `acquisition_seam` | Scanner tile seam, registration/cycle boundary, scan line, or stitching defect | Rescan/reprocess; source tissue may be intact |

Store these subtypes separately. A phase-1 model target named `crack` may be a
declared union, but only after the intended action and union members are frozen.
Never erase subtype labels during adjudication.

## 2. Annotation objects and required fields

Each artifact annotation is an instance with:

- `instance_id`: stable identifier unique within the image;
- `artifact_type`: one ontology value below;
- `geometry`: polygon/mask for affected area; line-like targets may additionally
  store a centerline polyline;
- `severity`: `minor`, `moderate`, or `severe` using Section 6;
- `confidence`: `high`, `medium`, or `low`;
- `reviewer_id` and timestamp;
- `channels_viewed`, magnification/scale, and z/cycle view if applicable;
- `action`: `none`, `mask`, `review`, `rescan/reprocess`, `recut/remount`, or
  `cannot_determine`;
- optional comment and linked confounder;
- adjudication state and final label.

The image manifest separately stores patient/block/slide/run/FOV identifiers,
modality, pixel spacing, coordinate transform, panel, channel names/roles,
instrument, site, batch, tissue, and valid/ignore regions.

Preferred delivery is a lossless raster label mask or GeoJSON polygons in level-0
coordinates, plus an immutable manifest. Record the rasterization rule. Do not use
anti-aliased masks or lossy JPEG labels. Instance IDs must survive raster export.

## 3. Label ontology

### 3.1 Primary targets

#### `fold`

Use when tissue is physically doubled, overlapped, rolled, wrinkled, creased, or
stacked. Common evidence includes locally doubled optical density/intensity,
compressed or superimposed nuclei, a ridge with paired edges, focal focus change,
and continuation of tissue morphology through the overlap.

Include the entire visibly affected folded/overlapped tissue, not only the darkest
ridge. When a fold causes a surrounding blur halo or shadow, label the fold core
as `fold` and the additional affected area as `out_of_focus`/`other_artifact`
unless the study explicitly defines a combined mask.

Do not label naturally dense tumor, lymphoid aggregates, cautery, hemorrhage,
pigment, necrosis, or high marker expression solely because it is dark or bright.

#### `tissue_tear`

Use for a physical discontinuity in an otherwise continuous section: split,
fissure, shattered tissue, torn edge, or missing corridor with displaced/irregular
tissue margins. Trace the visible affected corridor including damaged margins.
For a nearly zero-width line, annotate a centerline and a polygon at the visible
width; do not inflate it to make labeling easier.

Do not bridge a normal lumen, vessel, gland, adipose space, tissue edge, separation
between naturally disconnected fragments, or processing space without evidence of
a physical tear. If continuity cannot be determined, use `uncertain`.

#### `glass_crack`

Use when a line or branching network is attributable to glass/coverslip fracture
or crazing and affects the optical view of tissue. Evidence may include continuation
across tissue and background, refractive/double edges, branching geometry, or
focus/distortion inconsistent with tissue anatomy. Label the portion that affects
reviewable tissue/assay area; store the full visible line separately if useful for
remediation.

Do not infer `glass_crack` solely from a white line within tissue. If source-glass
inspection or another channel/view is needed and unavailable, label `uncertain`
rather than `tissue_tear`.

### 3.2 Secondary artifacts and confounders

These labels prevent a model from learning that every unusual structure is a fold
or crack. Whether they are model outputs or hard negatives is frozen in the study.

| Label | Definition |
|---|---|
| `knife_line` | Straight/repeated microtomy score, chatter, venetian-blind pattern, or blade-induced damage |
| `acquisition_seam` | Tile boundary, registration step, scan line, cycle shift, or stitching defect |
| `out_of_focus` | Tissue detail is not adequately resolved at the intended viewing scale |
| `air_bubble` | Mounting/assay bubble that obscures or distorts reviewable area |
| `foreign_material` | Dust, hair, debris, pen/marker, precipitate, or unrelated object |
| `antibody_aggregate` | Channel-specific punctate/clustered fluorescence not representing biology |
| `saturation_or_dropout` | Clipped signal, missing morphology, channel/cycle dropout, or detector failure |
| `registration_error` | Misalignment between structural channels, cycles, z planes, or tiles |
| `section_edge_damage` | Ragged/curling edge that does not establish an internal tear |
| `other_artifact` | Visible consequential artifact not represented above; comment is mandatory |

### 3.3 Hard negatives

Mark examples of target-like normal or non-target appearance as `hard_negative`
and choose a subtype when possible:

- vessel, gland, lumen, duct, fissure-like anatomy, alveolar or adipose space;
- tissue boundary, fragmentation gap, retraction space, or section hole;
- dense lymphoid/tumor region, hemorrhage, necrosis, mucus, pigment, cautery;
- bright or saturated normal marker expression, autofluorescence, cell aggregate;
- membrane network, sparse-cell region, morphology dropout not due to a tear;
- scan border, FOV edge, annotation boundary, text/label outside tissue.

Hard negatives remain valid negative pixels; they are not put in the ignore mask.
Their subtype is required for targeted error analysis.

### 3.4 `uncertain` and `ignore`

Use `uncertain` when the region is visible but artifact identity or target membership
cannot be established. Use `ignore` when no reliable decision is possible because
of missing imagery, corrupted tiles, severe unrelated artifact, inaccessible z/cycle,
annotation-tool limitation, or study-defined exclusion.

Do not use ignore to remove difficult false positives or ambiguous model errors
after predictions are seen. Ignore regions are determined without model output and
become the evaluator's `valid_mask` complement.

## 4. Drawing rules

1. Start at a low overview scale to understand tissue continuity, then annotate at
   the locked detail scale. Recheck at overview.
2. Draw the **affected area**, not just the most visually salient line. Do not add a
   fixed halo unless the protocol predefines one.
3. Keep distinct disconnected artifacts as separate instances. A continuous
   branching tear/glass crack is one instance unless physically interrupted.
4. Where two artifact types overlap, retain both labels in separate layers. The
   benchmark may derive a union later.
5. Stop a physical-artifact mask at the boundary of reviewable tissue unless the
   off-tissue continuation is required to establish subtype; store that evidence in
   an auxiliary layer.
6. Do not smooth away narrow branches or fill normal holes inside a fold.
7. Avoid annotating only easy/severe centers. Boundary-tolerant metrics account for
   edge uncertainty; partial positive masks create biased false-negative scoring.
8. Use native coordinates. If working at a pyramid level, the tool must export the
   exact level-to-level transform and nearest-neighbor rasterization.
9. Do not inspect model predictions during independent annotation.

### Boundary uncertainty

When the artifact core is certain but its affected edge is uncertain, draw the best
visible boundary and set confidence accordingly. Optionally add an uncertainty-band
layer. Do not silently expand `ignore` around every boundary. The annotation pilot
uses inter-reviewer distance to set the evaluation surface tolerance in physical
units.

### Thin tear/crack width

Trace the visible full width and, where supported, an ordered centerline. The
evaluator derives a deterministic skeleton from a raster mask, but a stored expert
centerline is valuable for review. Never force a minimum pixel width for the sake
of Dice. Record interruptions caused by missing evidence as uncertain segments.

## 5. Modality-specific viewing protocol

### H&E

- Review native RGB before any stain normalization.
- Use hematoxylin/eosin optical-density views only as aids.
- Compare low magnification for long folds/tears and higher magnification for
  doubled tissue, compressed nuclei, or microtomy detail.
- Do not call dense blue tissue a fold without structural evidence.
- If available, use glass-slide inspection or focus information to distinguish a
  coverslip crack from a tissue tear; record which evidence was used.

### COMET

- Begin with DAPI/nuclear and the locked autofluorescence/broad structural channels.
- Review individual channels as well as a standardized composite; a target must not
  be defined merely by a biological marker's high expression.
- Check cycle-to-cycle DAPI/registration views before calling a physical tear.
- Label antibody aggregates, saturation, cycle dropout, and registration errors
  separately.
- Record channel, cycle, and z context. A fold may affect all structural channels;
  a channel-specific spot is more likely an aggregate or acquisition issue.

### CosMx

- Use nuclear and broad membrane/tissue morphology channels with locked windowing;
  inspect z/union views when available.
- Do not use transcript density, cell type, or vendor cell masks to decide the
  primary optical artifact label. They may be assessed later as downstream impact.
- Check whether a line aligns with an FOV/tile boundary, missing morphology channel,
  z dropout, or registration seam before labeling a physical crack/tear.
- Record panel, channel roles/order, software version, FOV boundary, and z strategy.

If required channels or metadata are absent, mark the case for adjudication or
ignore according to the locked protocol; do not improvise a new viewing rule.

## 6. Severity and action

Severity combines extent with consequence. Area alone is insufficient: a narrow
tear through the only tumor region can matter more than a larger peripheral fold.

| Severity | Operational definition |
|---|---|
| `minor` | Visible artifact with negligible expected effect at intended use; no remediation, optional record/mask |
| `moderate` | Could alter review, segmentation, quantification, or usable area; requires human review or regional mask |
| `severe` | Obscures/loses critical tissue, causes material downstream error, or requires rescan, reprocess, recut, or remount |

Assign the expected action separately. During the pilot, reviewers document the
reason: critical-region involvement, affected tissue fraction, focus/occlusion,
cell/transcript impact, or inability to interpret. Adjudicators calibrate examples
into a severity atlas. The final definition is frozen before the locked test.

## 7. Confidence

| Confidence | Rule |
|---|---|
| `high` | Artifact type and extent are clear under the required viewing protocol |
| `medium` | Target is more likely than alternatives but edge or subtype has meaningful uncertainty |
| `low` | Evidence is insufficient for a reliable reference decision |

Low-confidence labels require adjudication and ordinarily become `uncertain` or
ignore if consensus cannot be reached. Confidence must not be increased simply to
meet positive-case quotas.

## 8. Independent review and adjudication

1. Two trained reviewers annotate each pilot and locked-test image independently,
   blinded to model output and to one another's result.
2. Compute image-level presence agreement, subtype confusion, severity/action
   agreement, Dice/surface distance, centerline distance, and instance matching.
3. Present disagreements to a qualified adjudicator with both annotations hidden
   or randomized where practical.
4. The adjudicator may accept one, create a new mask, change subtype/severity/action,
   or mark uncertain/ignore, always recording a reason code.
5. Freeze the adjudicated reference and checksum it before inference.

Suggested adjudication reason codes include `boundary_only`, `missed_instance`,
`subtype_confusion`, `normal_anatomy`, `severity_disagreement`, `channel_missing`,
`transform_error`, and `insufficient_evidence`.

Reviewers who helped tune the model may annotate development data, but locked-test
adjudication should include an independent domain owner. The final test must not be
re-labeled merely because model output appears plausible; suspected reference
errors follow a blinded change-control process.

## 9. Sampling requirements

The annotation queue must include, per modality:

- production-like consecutive/random samples to estimate prevalence;
- folds and each approved crack subtype across minor, moderate, and severe levels;
- clean images and explicit hard negatives;
- multiple tissues, sites/operators, devices, batches/lots, panels, and time periods;
- missing/degraded channels and acquisition failures;
- repeated/adjacent images kept within one data split;
- enough severe positives to estimate sensitivity with the planned confidence bound.

Do not sample only from obvious QC failures. Enrichment is labeled in the manifest
and analyzed separately from the prevalence cohort.

## 10. Annotation quality checks

Before freeze, an independent script verifies:

- every image and instance has required identifiers and ontology values;
- label arrays align with image shape, orientation, level, origin, and spacing;
- masks contain integer labels only and instance IDs are unique;
- polygons are valid and rasterize deterministically;
- target masks lie inside the declared valid/reference area unless justified;
- `ignore` does not overlap scored labels without a documented precedence rule;
- severity/action/confidence combinations are logically valid;
- duplicated patient/block/slide/run units do not cross splits;
- a seeded visual sample plus every severe case receives overlay review;
- reference files and manifests have immutable checksums.

Any transform error invalidates the affected reference until corrected and re-frozen.

## 11. Synthetic and public labels

Synthetic masks may exercise ontology fields, transforms, thin-line metrics, and
reporting, but synthetic samples are never mixed into inter-reviewer agreement,
acceptance, or generalization estimates.

Public labels are mapped rather than silently renamed. For example:

- GrandQC `fold` can map to `fold`, but its other classes do not create a crack label;
- Foucart `Tear&Fold` remains a conflated source label and cannot establish either
  subtype independently;
- HistoArtifacts `damaged_tissue` is not automatically `tissue_tear`;
- QUALIFAI `external_artifact` is not automatically glass crack or tissue tear.

Record original label, mapped label, mapping rationale, dataset/version, license,
and exclusions. Public access does not imply permission for corporate model
development or redistribution.

## 12. Reference freeze checklist

- [ ] The intended action and exact `crack` union are approved.
- [ ] Reviewers completed training and calibration examples.
- [ ] Two independent annotations and adjudication are complete.
- [ ] Hard negatives and all required generalization strata are represented.
- [ ] Valid/ignore masks were defined without model output.
- [ ] Pixel spacing, coordinate transform, channel roles, and viewing protocol are fixed.
- [ ] Severity/action rules and a visual atlas are approved.
- [ ] Split leakage and identifier privacy checks pass.
- [ ] Dataset/model license and provenance review is documented.
- [ ] Manifests, masks, and protocol have checksums and versioned immutable storage.
- [ ] Thresholds and metric tolerances will be selected only on development data.
