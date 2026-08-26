# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Added `BinClass.compile_document_predictor/2` for scanning one complete text
  with overlapping token windows. It returns every chunk's exact original byte
  span, token count, validation status, and raw prediction without truncating,
  normalizing, aggregating, or grouping documents.
- Added the public `BinClass.Evaluation` API for fixed-threshold metrics, exact
  threshold calibration, and selection among named probability strategies.
  Evaluation supports arbitrary row accessors, soft Brier targets, eligibility
  filtering, multiple objectives, and explicit false-positive cost.
- Added explicit `:decision_policy` training configuration. Classification now
  supports argmax, fixed positive thresholds, and opt-in validation threshold
  optimization using accuracy, balanced accuracy, or F1 with an optional
  false-positive cost.
- Added persisted `:minimum_input_tokens` validation. Short inputs now return
  `:insufficient_input` while retaining their raw class probabilities.
- Added opt-in soft training targets, accepting positive-class probabilities
  from `0.0` through `1.0` and converting them to `[1 - p, p]` distributions.
- Added configurable target boundaries for balancing, validation metrics, and
  decision-threshold calibration.
- Added serialized target-mode metadata.

### Changed

- Extracted compiled prediction and shared model configuration into focused
  internal modules, removing runtime dependency cycles.
- Public APIs now validate known option values before training, loading, model
  construction, or serving work begins.
- Serialization and every inference path now share one complete classifier
  validation contract, including training metadata and tokenizer metadata.
- Serving and compiled prediction now accept only a string or a non-empty list
  of strings and report invalid input with the same error.
- Argmax is now the default decision policy for every architecture, and `:cnn`
  is the default architecture.
- Decision policies and input validation no longer derive from model
  architecture. Serving and compiled prediction use the same persisted policy.
- Serialized classifiers now require a tagged decision policy and an explicit
  `minimum_input_tokens` value. Classifiers using the earlier policy schema must
  be retrained.
- Output labels must now be two distinct binaries and default to `["0", "1"]`,
  preventing custom label atoms from making safe deserialization fail in a
  fresh VM.
- Vector lengths are now validated against each architecture's convolution
  requirements before tokenizer or model setup.
- Threshold optimization excludes inputs that fail minimum-token validation
  and requires eligible validation examples from both classes.
- Removed the model-side short-input negative bias from the Transformer graph.
- Classifier artifacts now contain a required model-graph revision. Artifacts
  without the revision predate the exact padding semantics and must be
  retrained rather than silently running against a different graph.
- Training accepts an optional seed for deterministic data ordering, model
  initialization, and dropout state.
- Training and validation partitions are now stratified whenever both target
  classes have enough examples to occur in both partitions. Threshold
  optimization fails before tokenizer or model setup when that is impossible.
- Public functions that accept keyword options now reject unknown keys instead
  of silently ignoring them, including compiled prediction, tokenizer training,
  and direct model construction.
- Threshold calibration now computes validation predictions once and reuses
  them across all threshold candidates.
- Serving and compiled prediction now obtain active lengths directly from the
  truncated tokenizer encoding instead of rescanning padded ID vectors.
- Model graphs now share one token mask and active-length node across embedding
  masking, pooling, and attention.
- All architectures now mask padded embeddings and exclude padded positions
  from global pooling. The Transformer also masks padded attention keys.
- Training now sizes model embeddings from the tokenizer's actual vocabulary
  instead of the requested maximum vocabulary size.
- Vectorization, model construction, validation, serving, and compiled
  prediction now use the tokenizer's actual padding-token ID.
- Serialized classifiers now store and validate `vocab_size`, `pad_token_id`,
  and `unk_token_id` against the embedded tokenizer JSON. Classifiers written
  with the previous schema must be retrained.
- Added `BinClass.Tokenizer.metadata/1` and changed the vectorizer's
  three-argument `build` function to `BinClass.Vectorizer.build/4`.
- Renamed the classifier field, training and serving option, and serialized key
  from `model_version` to `architecture`.
- Serialized classifiers now require an explicit named `architecture`. Existing
  binaries using the old schema must be retrained.

### Removed

- Removed the redundant `:conservative_cnn` architecture. Use `:cnn` with an
  explicit `:decision_policy` and `:minimum_input_tokens` configuration instead.
- Removed the built-in `:tune` training option and its learning-rate/dropout
  grid search. Supply explicit `:learning_rate` and `:dropout_rate` values or
  run external experiments instead.
- Removed the `:vectorization_concurrency` and `:threshold_candidates` training
  options. Vectorization concurrency is selected automatically, and threshold
  calibration now evaluates every distinct decision partition in the
  validation predictions and centers thresholds between adjacent probabilities.
- Removed the `:calibrate_threshold` and `:false_positive_penalty` training
  options. Threshold optimization is now requested explicitly through
  `:decision_policy`.
- Removed the `:positive_threshold` and `:min_positive_tokens` serving
  overrides. Inference now always uses the classifier's persisted policy.
- Removed the unused Explorer dependency and its test-only tensor conversion
  helper.
- Removed legacy deserialization defaults. Saved classifiers must now contain
  the complete current schema.
- Removed the test-only tokenizer default-vocabulary accessor and redundant
  six-argument training split wrapper.
- Removed tokenizer metadata overrides from the low-level serving constructor;
  serving now always derives vocabulary size and special-token IDs from the tokenizer.
- Replaced the low-level public serving constructor with `BinClass.serving/2`,
  which accepts a complete classifier and only runtime serving options. The
  implementation module is now internal and has no classifier-option defaults.
- Removed the implicit current model revision from manually constructed
  classifier structs. A revision must now be present explicitly.
- Removed numeric model architecture identifiers. Training, serving, serialization,
  and deserialization now require named architectures such as `:cnn`.
- Removed all fallback handling for the previous model-version schema.
- Removed serialization and inference fallbacks for missing classifier dropout
  and target metadata. Incomplete classifiers now raise a clear error.

### Fixed

- Predictions no longer change when the same token sequence receives a
  different amount of right-padding.
- Padding no longer uses the unknown-token ID, so unknown tokens remain model
  input and count toward minimum-input validation.
- Active-token counting now uses the exact non-padding token IDs consistently
  in training, serving, compiled prediction, and document prediction. Literal
  `[PAD]` input can no longer bypass minimum-token validation.
- Global pooling now uses exact token masks, preserving internal padding holes.
  Valid-convolution architectures propagate kernel-specific masks so outputs
  whose receptive fields include padding cannot affect predictions.
- Prediction and document-chunk results now expose `:active_token_count`.

## [0.2.1] - 2026-08-20

### Changed

- Updated Nx and EXLA to `0.13.1`.

### Validation

- Verified the complete test suite and one-epoch EXLA training with all seven supplied `.bin` training fixtures.

## [0.2.0] - 2026-06-23

### Added

- Added model version 7, `:conservative_cnn`, a conservative CNN architecture intended to reduce false positives.
- Added false-positive-aware validation scoring for checkpoint selection and hyperparameter tuning.
- Added threshold calibration for v7, including persisted `decision_policy` metadata in serialized classifiers.
- Added v7 inference safeguards for low-signal inputs through configurable positive thresholds and minimum positive token counts.

### Changed

- Changed the default training model to v7, `:conservative_cnn`.
- Changed the default vector length to `512`.
- Updated serialization, loading, serving, and compiled prediction paths to reuse persisted decision policies when available.

### Documentation

- Updated README and generated documentation sources for v7, threshold calibration, production guidance, model versions, and the `0.2.0` dependency snippet.

## [0.1.3] - 2026-06-15

### Added

- Added model version 6, `:transformer`, with sequence-length logit bias for short or missing text inputs.
- Added static sinusoidal positional embeddings to the transformer model.
- Added transformer execution and logit-bias tests.

### Changed

- Updated dependencies, including Nx `0.12.1` and ExDoc `0.40.3`.
- Reduced JIT recompilation warnings during training and hyperparameter tuning.

### Documentation

- Updated release documentation for the transformer model.

## [0.1.2] - 2026-05-17

### Changed

- Updated dependency versions.
- Fixed the Nx dependency version.

## [0.1.1] - 2026-04-19

### Added

- Added model version 5 with a multi-scale architecture and made it the default at the time.
- Added tests for model version 5.

### Changed

- Refactored model architecture names to be more descriptive.
- Updated CI to newer Elixir and OTP versions.

### Fixed

- Fixed Axon deprecation warnings by using `Axon.ModelState.empty()`.
- Fixed compiler warnings and formatted tests.

## [0.1.0] - 2026-02-06

### Added

- Added the initial binary text classification library built on Axon and Nx.
- Added tokenizer, vectorizer, training, serving, saving, and loading workflows.
- Added explicit label mapping support.
- Added configurable compiler/backend options for training and serving.
- Added model versioning for backward-compatible serialized classifiers.
- Added model versions 1 through 4, including CNN, mixed-pooling CNN, multi-scale CNN, and Sep-SE-CNN.
- Added hyperparameter auto-tuning for learning rate and dropout.
- Added `BinClass.compile_predictor/2` for low-latency in-process inference.
- Added examples for training, serving, configurable backends, and simple inference.
- Added Hex package metadata, MIT license, CI workflow, and Dependabot configuration.

### Changed

- Optimized model architecture and training performance before the first tagged release.
- Reverted the experimental v5 work before `v0.1.0` and kept v4 as the default at release time.
- Marked the temporary-file helper module as internal.

### Documentation

- Added and refined README, examples documentation, and module documentation for the initial Hex release.

[Unreleased]: https://github.com/preciz/bin_class/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/preciz/bin_class/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/preciz/bin_class/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/preciz/bin_class/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/preciz/bin_class/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/preciz/bin_class/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/preciz/bin_class/releases/tag/v0.1.0
