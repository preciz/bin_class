# BinClass

[![test](https://github.com/preciz/bin_class/actions/workflows/test.yml/badge.svg)](https://github.com/preciz/bin_class/actions/workflows/test.yml)

An easy-to-use Elixir library for building, training, and deploying binary text classifiers with [Axon](https://github.com/elixir-nx/axon).

This library provides a simplified interface for training a neural network on text data and using it for predictions, handling tokenization, vectorization, and model training out of the box.

## Installation

The package can be installed by adding `bin_class` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:bin_class, "~> 0.2.0"}
  ]
end
```

## Quick Start

### 1. Prepare your data

Data should be an enumerable of maps containing `:text` and `:label` (0 or 1).

```elixir
data = [
  %{text: "This is a great product!", label: 1},
  %{text: "I really hated this experience.", label: 0},
  # ... more samples
]
```

### 2. Train the model

```elixir
# Labels can be a list (index 0 and 1) or an explicit map.
# The default model is v7, a conservative CNN optimized to reduce false positives.
classifier = BinClass.Trainer.train(data,
  epochs: 10,
  labels: %{0 => :negative, 1 => :positive}
)

# Optional: enable auto-tuning to search learning rate and dropout.
# For production training, tune: false is usually preferred unless you have
# a stable validation split and enough data.
classifier = BinClass.Trainer.train(data, tune: true)
```

By default, training uses a fixed token vector length of `512`.

### 3. Save and Load

You can save the entire model (including tokenizer, parameters, decision policy, and metadata) to a single file.

```elixir
BinClass.save(classifier, "my_model.bin")

# Load the model as an Nx.Serving struct (recommended for most apps)
serving = BinClass.load("my_model.bin")
```

### 4. Optimized Inference

There are two ways to run predictions:

#### A. Using `Nx.Serving` (High Throughput)
Recommended for web servers and concurrent applications. It handles automatic batching.

```elixir
prediction = Nx.Serving.run(serving, "I love this library!")
```

#### B. Using a Compiled Predictor (Ultra-Low Latency)
Recommended for CLI tools or scripts where you want the lowest possible latency for single items by bypassing the serving overhead.

```elixir
classifier = BinClass.load_classifier("my_model.bin")
predict = BinClass.compile_predictor(classifier)

result = predict.("This is ultra fast.")
```

## Examples

Check out the `examples/` directory for scripts demonstrating various use cases:

- `simple_inference.exs`: Shows how to quickly run predictions with a pre-trained model.
- `train_and_save.exs`: Demonstrates the full workflow of training a model and saving it to disk.
- `production_serving.exs`: Illustrates how to integrate `BinClass` into a supervision tree for production environments.
- `configurable_backend.exs`: Shows how to use custom compilers and definition options for training and inference.

## Features

- **Production Ready**: Built on `Nx.Serving` for automatic batching and process isolation.
- **Unified Serialization**: Save and load the entire classifier state, including tokenizer, model parameters, and calibrated decision policy, from a single file.
- **Model Versioning**: Decouples model parameters from code changes by explicitly versioning architectures.
- **Multiple Architectures**: Supports legacy CNN variants, **Sep-SE-CNN** (Separable Convolutions + Squeeze-and-Excitation), **Transformer Encoder**, and v7 **Conservative CNN**.
- **Conservative v7 Default**: Uses a v1-style CNN backbone with false-positive-aware checkpoint selection, low-signal safeguards, and persisted threshold calibration.
- **Early Stopping**: Automatically halts training when validation loss stops improving.
- **Automatic Class Balancing**: Handles imbalanced datasets via automated oversampling.
- **Automated Tokenization**: Automatically builds vocabulary from training data or accepts custom streams.
- **Efficient**: Uses `EXLA` as the default compiler for high-performance training and inference, with support for other `Nx` backends and compilers.

## Model Versions

Models are versioned so saved classifiers keep loading even when the library default changes:

- `1` / `:cnn`: original CNN
- `2` / `:cnn_mixed_pooling`: CNN with mixed pooling
- `3` / `:multi_scale_cnn`: multi-scale CNN
- `4` / `:sep_se_cnn`: separable CNN with squeeze-and-excitation
- `5` / `:parallel_cnn`: parallel CNN
- `6` / `:transformer`: transformer encoder
- `7` / `:conservative_cnn`: conservative CNN default

v7 is based on the original simple CNN backbone, with production behavior tuned for cases where false positives are more costly than false negatives. It selects checkpoints using a false-positive penalty and stores a calibrated positive threshold in the serialized classifier.

## Production Notes

For production training, start with the defaults unless you have a specific validation strategy:

```elixir
classifier = BinClass.Trainer.train(data,
  epochs: 10,
  tune: false,
  model_version: :conservative_cnn,
  vector_length: 512,
  false_positive_penalty: 0.5,
  calibrate_threshold: true
)
```

The calibrated `decision_policy` is saved with the model and reused by both `BinClass.load/2` and `BinClass.compile_predictor/2`. Older serialized models that do not include this field still load normally and fall back to their version defaults.
