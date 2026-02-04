# BinClass

An Elixir library for building and using binary text classifiers using [Axon](https://github.com/elixir-nx/axon).

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
# By default, uses labels [0, 1]
# Returns a %BinClass.Classifier{} struct
classifier = BinClass.Trainer.train(data, epochs: 10, labels: [:negative, :positive])
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

## Features

- **Production Ready**: Built on `Nx.Serving` for automatic batching and process isolation.
- **Unified Serialization**: Save and load the entire classifier state from a single file.
- **KimCNN Architecture**: Uses multi-scale convolutions (kernels 3, 4, 5) to capture varying n-gram lengths.
- **Early Stopping**: Automatically halts training when validation loss stops improving.
- **Automatic Class Balancing**: Handles imbalanced datasets via automated oversampling.
- **Automated Tokenization**: Automatically builds vocabulary from training data or accepts custom streams.
- **Efficient**: Uses `EXLA` as the default compiler for high-performance training and inference.
