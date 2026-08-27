# BinClass

[![test](https://github.com/preciz/bin_class/actions/workflows/test.yml/badge.svg)](https://github.com/preciz/bin_class/actions/workflows/test.yml)

An easy-to-use Elixir library for building, training, and deploying binary text classifiers with [Axon](https://github.com/elixir-nx/axon).

This library provides a simplified interface for training a neural network on text data and using it for predictions, handling tokenization, vectorization, and model training out of the box.

## Installation

The package can be installed by adding `bin_class` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:bin_class, "~> 0.3.0"}
  ]
end
```

## Quick Start

### 1. Prepare your data

Data can be an enumerable of maps (`%{text: "...", label: 1}` or `%{"text" => "...", "label" => 1}`) or tuples (`{"...", label}`).

```elixir
data = [
  %{text: "This is a great product!", label: 1},
  %{text: "I really hated this experience.", label: 0},
  # ... more samples
]

# Inspect dataset balance and text length statistics:
summary = BinClass.summary(data)
IO.puts(BinClass.format_summary(summary))

# Stratified train / validation split:
{train_data, val_data} = BinClass.Dataset.split(data, validation_split: 0.1, seed: 42)
```

### 2. Train the model

```elixir
# Output labels can be customized with a two-element list or map.
classifier = BinClass.train(train_data,
  epochs: 5,
  labels: %{0 => "negative", 1 => "positive"},
  label_smoothing: 0.02
)
```

By default, training uses the **`:parallel_cnn`** architecture with 2-head attentive pooling, a fixed sequence length of `512`, NFKC lowercasing normalizer, vocabulary size `8_000`, and argmax classification.

### 3. Evaluate the model

```elixir
result = BinClass.evaluate(classifier, val_data)
IO.puts(BinClass.format_report(result))
```

This outputs a comprehensive evaluation report with **Accuracy**, **Balanced Accuracy**, **Precision**, **Recall**, **Specificity**, **F1 Score**, **Cross-Entropy Loss**, and the full **Confusion Matrix**.

### Soft targets

Training labels may also represent a graded positive-class probability. Enable
soft targets explicitly and provide values from `0.0` through `1.0`:

```elixir
data = [
  %{text: "Clearly negative example", label: 0.0},
  %{text: "Weak positive evidence", label: 0.2},
  %{text: "Strong positive evidence", label: 0.9},
  %{text: "Clearly positive example", label: 1.0}
]

classifier = BinClass.Trainer.train(data,
  target_mode: :soft,
  target_threshold: 0.5
)
```

Each value `p` is trained as the distribution `[1 - p, p]`. The target
threshold is used only to derive binary classes for balancing and validation
metrics. Hard `0`/`1` targets remain the default. Target settings are retained
as training metadata when the classifier is serialized.

### Decision policies and input validation

Classification defaults to argmax. Decision behavior is explicit and does not
depend on model architecture. A fixed positive threshold can be selected with:

```elixir
classifier = BinClass.Trainer.train(data,
  decision_policy: {:positive_threshold, 0.7}
)
```

Threshold optimization is opt-in and uses the validation partition. Supported
objectives are `:accuracy`, `:balanced_accuracy`, and `:f1`. An optional
false-positive cost is subtracted from the objective score as a multiple of the
false-positive rate. Optimization requires at least one eligible validation
example from each class:

```elixir
classifier = BinClass.Trainer.train(data,
  decision_policy:
    {:optimize_threshold,
     [objective: :balanced_accuracy, false_positive_cost: 0.75]}
)
```

The selected threshold and optimization metadata are persisted with the
classifier. To abstain from classifying short inputs, configure input validation
separately:

```elixir
classifier = BinClass.Trainer.train(data,
  minimum_input_tokens: 64
)
```

An input below the minimum returns `label: :insufficient_input` and
`confidence: nil`. Its raw class probabilities remain available in the
`:probabilities` map. Every result also reports `:active_token_count`, defined
as the number of truncated token IDs other than the tokenizer's padding ID.
Literal `[PAD]` tokens therefore cannot satisfy the minimum. Ineligible inputs
are excluded from threshold optimization and reported validation accuracy.

### Evaluation and calibration

`BinClass.Evaluation` can also evaluate probabilities produced by external or
multi-stage inference pipelines. Fixed-threshold evaluation returns confusion
counts, accuracy, precision, recall, specificity, false-positive rate,
balanced accuracy, F1, Brier score, and the configured objective score:

```elixir
metrics =
  BinClass.Evaluation.evaluate(rows,
    threshold: 0.7,
    probability: :positive_probability,
    label: :positive?,
    target: :soft_target,
    eligible: :eligible?
  )
```

The probability, label, soft target, and eligibility selectors may be map keys
or one-argument functions. The `:target` and `:eligible` selectors are optional.

Use `calibrate/2` to inspect every distinct decision boundary represented by
the observed probabilities:

```elixir
metrics =
  BinClass.Evaluation.calibrate(rows,
    objective: :balanced_accuracy,
    false_positive_cost: 0.75
  )
```

Named probability strategies can be calibrated and compared in one call:

```elixir
selection =
  BinClass.Evaluation.select(rows,
    strategies: [
      maximum: & &1.probabilities.maximum,
      top_two_mean: & &1.probabilities.top_two_mean
    ],
    label: :label,
    false_positive_cost: 0.75
  )

selection.strategy
#=> :top_two_mean
```

### 4. Save and Load

You can save the entire model (including tokenizer, parameters, decision policy, and metadata) to a single file.

```elixir
BinClass.save(classifier, "my_model.bin")

# Load the model as an Nx.Serving struct (recommended for most apps)
serving = BinClass.load("my_model.bin")

# Or build a serving from an existing classifier without serializing it
serving = BinClass.serving(classifier, batch_size: 32)
```

### 5. Optimized Inference

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

#### C. Inspecting a Complete Document

Compile a document predictor when an input may exceed the classifier's fixed
vector length. It scans the complete string using overlapping token windows and
returns every chunk with its exact byte span and raw prediction:

```elixir
classifier = BinClass.load_classifier("my_model.bin")

predict_document =
  BinClass.compile_document_predictor(classifier,
    chunk_overlap: 128,
    batch_size: 32
  )

%{status: :ok, chunks: chunks} = predict_document.(large_document)

Enum.each(chunks, fn chunk ->
  IO.inspect({
    chunk.start_byte,
    chunk.end_byte,
    chunk.token_count,
    chunk.active_token_count,
    chunk.prediction.probabilities
  })
end)
```

The predictor accepts one document at a time. It does not normalize, truncate,
aggregate, or apply a separate document-level decision. Byte offsets refer to
the exact input string, with an inclusive start and exclusive end. A chunk's
`:token_count` is its encoded size, while `:active_token_count` excludes padding
special tokens and controls minimum-input validation. Individual chunks that
fail validation retain their raw probabilities.

## Examples

Check out the `examples/` directory for scripts demonstrating various use cases:

- `simple_inference.exs`: Shows how to quickly run predictions with a pre-trained model.
- `train_and_save.exs`: Demonstrates the full workflow of training a model and saving it to disk.
- `production_serving.exs`: Illustrates how to integrate `BinClass` into a supervision tree for production environments.
- `configurable_backend.exs`: Shows how to use custom compilers and definition options for training and inference.

## Public API

The supported API is organized around these modules:

| Module | Purpose |
|---|---|
| `BinClass` | Primary entry point: train, evaluate, summarize, format reports, save/load classifiers, build servings and compiled predictors. |
| `BinClass.Dataset` | Inspect dataset distributions and word/character statistics (`summary/1`, `format_summary/1`), and perform stratified train/val/test splits (`split/2`). |
| `BinClass.Trainer` | Train complete `BinClass.Classifier` artifacts from labeled text. |
| `BinClass.Evaluation` | Evaluate trained classifiers on datasets, compute confusion matrices, calibrate decision thresholds, or compare probability strategies. |
| `BinClass.Model` | List and directly build the supported Axon neural architectures. |
| `BinClass.Tokenizer` | Train tokenizers and read the metadata needed for vectorization. |
| `BinClass.Vectorizer` | Encode text into fixed-length padded token vectors. |
| `BinClass.Classifier` | The complete trained classifier artifact consumed by persistence and inference APIs. |

Modules hidden from the generated documentation are implementation details and
may change without notice. Use the `BinClass` entry points for persistence and
inference instead of calling serialization, serving, or predictor internals
directly.

## Features

- **Production Ready**: Built on `Nx.Serving` for automatic batching and process isolation.
- **Top-Level Convenience**: One-line training (`BinClass.train/2`), dataset inspection (`BinClass.summary/1`), and comprehensive evaluation (`BinClass.evaluate/3`).
- **Unified Serialization**: Save and load the entire classifier state, including tokenizer, model parameters, explicit decision policy, input validation, and training metadata, from a single file.
- **Named Architectures**: Serialized classifiers identify their architecture explicitly by name.
- **Multiple Architectures**: Supports **Parallel CNN with Attentive Pooling** (default), CNN variants, **Sep-SE-CNN** (Separable Convolutions + Squeeze-and-Excitation), and **Transformer Encoder**.
- **Target Regularization**: Supports `:label_smoothing` to improve calibration and prevent overconfidence on ambiguous boundary labels.
- **Explicit Decisions**: Defaults to argmax and supports fixed or validation-optimized positive thresholds independently of architecture.
- **Input Abstention**: Optionally returns `:insufficient_input` for inputs below a configured token minimum while retaining raw probabilities.
- **Complete-Document Inspection**: Scans arbitrarily long text into overlapping token windows and returns every chunk's exact input byte span and raw prediction.
- **Early Stopping**: Automatically halts training when validation loss stops improving.
- **Automatic Class Balancing**: Handles imbalanced datasets via automated oversampling.
- **Automated Tokenization**: Automatically builds vocabulary from training data or accepts custom streams.
- **Efficient**: Uses `EXLA` as the default compiler for high-performance training and inference, with support for other `Nx` backends and compilers.

## Model Architectures

Classifiers store one of these named architectures:

- `:parallel_cnn`: parallel CNN with multi-scale kernel convolutions and multi-head attentive pooling (**default**)
- `:dense_dilated_cnn`: dense-connected multi-scale dilated CNN with 4-head attentive pooling
- `:cnn`: original CNN
- `:cnn_mixed_pooling`: CNN with mixed pooling
- `:multi_scale_cnn`: multi-scale CNN
- `:sep_se_cnn`: separable CNN with squeeze-and-excitation
- `:transformer`: transformer encoder

No architecture implicitly changes label thresholds or rejects short inputs.
Those behaviors are configured through `:decision_policy` and
`:minimum_input_tokens`.

Numeric architecture identifiers are not accepted. Serialized classifiers must
contain one of the names above.

## Production Notes

For production training, start with argmax unless validation results justify a
different operating point:

```elixir
classifier = BinClass.train(data,
  epochs: 5,
  architecture: :parallel_cnn,
  vector_length: 512,
  decision_policy:
    {:optimize_threshold,
     [objective: :balanced_accuracy, false_positive_cost: 0.5]},
  label_smoothing: 0.02,
  minimum_input_tokens: 64
)
```

The explicit `decision_policy` and `minimum_input_tokens` are saved with the
model and reused identically by `BinClass.load/2` and
`BinClass.compile_predictor/2`. Raw probabilities are returned for every input,
including inputs that fail minimum-length validation.
