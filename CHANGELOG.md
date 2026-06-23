# Changelog

All notable changes to this project are documented in this file.

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

[0.2.0]: https://github.com/preciz/bin_class/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/preciz/bin_class/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/preciz/bin_class/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/preciz/bin_class/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/preciz/bin_class/releases/tag/v0.1.0
