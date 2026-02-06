# BinClass

[![test](https://github.com/preciz/bin_class/actions/workflows/test.yml/badge.svg)](https://github.com/preciz/bin_class/actions/workflows/test.yml)

An easy-to-use Elixir library for building, training, and deploying binary text classifiers with [Axon](https://github.com/elixir-nx/axon).

This library provides a simplified interface for training a neural network on text data and using it for predictions, handling tokenization, vectorization, and model training out of the box.

## Installation

The package can be installed by adding `bin_class` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:bin_class, "~> 0.1.0"}
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
# Labels can be a list (index 0 and 1) or an explicit map
classifier = BinClass.Trainer.train(data,
  epochs: 10,
  labels: %{0 => :negative, 1 => :positive}
)

# Optional: Enable auto-tuning to find the best Learning Rate and Dropout
classifier = BinClass.Trainer.train(data, tune: true)
```

### 3. Save and Load

You can save the entire model (including tokenizer, parameters, and metadata) to a single file.

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
- **Unified Serialization**: Save and load the entire classifier state from a single file.
- **Model Versioning**: Decouples model parameters from code changes by explicitly versioning architectures.
- **CNN Architecture**: Uses **Sep-SE-CNN** (Separable Convolutions + Squeeze-and-Excitation) with multi-scale kernels (3, 4, 5-gram) and mixed pooling for state-of-the-art efficiency and accuracy.
- **Early Stopping**: Automatically halts training when validation loss stops improving.
- **Automatic Class Balancing**: Handles imbalanced datasets via automated oversampling.
- **Automated Tokenization**: Automatically builds vocabulary from training data or accepts custom streams.
- **Efficient**: Uses `EXLA` as the default compiler for high-performance training and inference, with support for other `Nx` backends and compilers.
