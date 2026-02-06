# BinClass

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
```

### 3. Save and Load

You can save the entire model (including tokenizer, parameters, and metadata) to a single file.

```elixir
BinClass.save(classifier, "my_model.bin")

# Load the model as an Nx.Serving struct
serving = BinClass.load("my_model.bin")
```

### 4. Inference

Run predictions using `Nx.Serving`. This handles batching automatically for high throughput.

```elixir
# Single prediction
prediction = Nx.Serving.run(serving, "I love this library!")
# %{
#   label: :positive,
#   confidence: 0.99,
#   probabilities: %{negative: 0.01, positive: 0.99}
# }

# Batch prediction
results = Nx.Serving.run(serving, ["Great tool", "Bad bugs"])
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
- **CNN Architecture**: Uses 1D convolutions with mixed global max and average pooling for robust feature extraction.
- **Early Stopping**: Automatically halts training when validation loss stops improving.
- **Automatic Class Balancing**: Handles imbalanced datasets via automated oversampling.
- **Automated Tokenization**: Automatically builds vocabulary from training data or accepts custom streams.
- **Efficient**: Uses `EXLA` as the default compiler for high-performance training and inference, with support for other `Nx` backends and compilers.
