defmodule BinClass do
  @moduledoc """
  A reusable library for building, training, and using binary classifiers using Axon.

  ## Usage

  ### Training

      data = [
        %{text: "This is a positive text", label: 1},
        %{text: "This is a negative text", label: 0},
        # ... more data
      ]

      # Use a map for explicit label mapping
      classifier = BinClass.train(data,
        epochs: 5,
        labels: %{0 => "negative", 1 => "positive"}
      )

      # Prediction uses argmax unless an explicit decision policy was selected
      # during training.

      # Save the model
      BinClass.save(classifier, "model.bin")

  ### Prediction

      # Load the model as a serving (supports custom compiler/defn_options)
      serving = BinClass.load("model.bin", compiler: EXLA)

      result = Nx.Serving.run(serving, "Some text to classify")
      # result is %{label: "positive", confidence: 0.99, active_token_count: 4, ...}

  ### Ultra-Low Latency Inference

      # Load raw classifier and compile a predictor function
      classifier = BinClass.load_classifier("model.bin")
      predict = BinClass.compile_predictor(classifier)

      result = predict.("Instant prediction")

  """

  @doc """
  Trains a binary classifier.

  Delegates to `BinClass.Trainer.train/2`. See `BinClass.Trainer.train/2` for full option details.
  """
  defdelegate train(data, opts \\ []), to: BinClass.Trainer

  @doc """
  Saves the classifier to a file.
  """
  def save(%BinClass.Classifier{} = classifier, path) do
    binary = serialize(classifier)
    File.write!(path, binary)
  end

  @doc """
  Serializes the classifier to a binary.

  The serialized data includes tokenizer state, its exact vocabulary and
  special-token IDs, model parameters, labels, vector length, named model
  architecture and graph revision, training metadata (including target mode
  and threshold), and the explicit decision policy and minimum-input
  validation setting.
  """
  def serialize(%BinClass.Classifier{} = classifier) do
    BinClass.Serialization.serialize(classifier)
  end

  @doc """
  Loads a saved model from a file and returns an Nx.Serving struct.
  """
  def load(path, opts \\ []) do
    opts = validate_serving_options!(opts)

    path
    |> load_classifier()
    |> build_serving(opts)
  end

  @doc """
  Deserializes a saved model from a binary and returns an Nx.Serving struct.
  """
  def deserialize(binary, opts \\ []) when is_binary(binary) do
    opts = validate_serving_options!(opts)

    binary
    |> deserialize_classifier()
    |> build_serving(opts)
  end

  @doc """
  Builds an `Nx.Serving` from a classifier.

  Model architecture, tokenizer metadata, labels, vector length, decision
  policy, and minimum-input validation always come from the classifier.

  ## Options

    * `:compiler` - A module implementing `Nx.Defn.Compiler`. Defaults to `EXLA`.
    * `:batch_size` - The positive serving batch size. Defaults to `16`.
    * `:defn_options` - A keyword list passed to the compiled definition. Defaults to `[]`.
  """
  def serving(%BinClass.Classifier{} = classifier, opts \\ []) do
    opts = validate_serving_options!(opts)
    build_serving(classifier, opts)
  end

  defp build_serving(classifier, opts) do
    classifier = BinClass.Classifier.validate!(classifier)

    serving_opts =
      Keyword.merge(opts,
        vector_length: classifier.vector_length,
        labels: classifier.labels,
        architecture: classifier.architecture,
        dropout_rate: classifier.dropout_rate,
        decision_policy: classifier.decision_policy,
        minimum_input_tokens: classifier.minimum_input_tokens
      )

    BinClass.Serving.new(classifier.model_params, classifier.tokenizer, serving_opts)
  end

  @doc """
  Loads a saved model from a file and returns a BinClass.Classifier struct.
  """
  def load_classifier(path) do
    binary = File.read!(path)
    deserialize_classifier(binary)
  end

  @doc """
  Deserializes a saved model from a binary and returns a BinClass.Classifier struct.

  Serialized classifiers must contain the complete current schema, including
  an explicit decision policy and minimum-input validation setting. The stored
  vocabulary size, padding ID, and unknown-token ID must match the tokenizer JSON.
  """
  def deserialize_classifier(binary) when is_binary(binary) do
    BinClass.Serialization.deserialize(binary)
  end

  @doc """
  Compiles the classifier into a highly optimized, in-process prediction function.

  This is intended for scenarios where lowest possible latency is required
  and batching (provided by Nx.Serving) is not necessary (e.g. CLI tools,
  single-user scripts, or very low-concurrency high-speed inference).

  Returns an anonymous function that takes a text (string) or a non-empty list
  of texts and returns the classification results. Inputs below the classifier's
  `minimum_input_tokens` return `label: :insufficient_input` and
  `confidence: nil`, while retaining raw class probabilities. Every result
  includes `:active_token_count`, defined consistently as the number of
  truncated token IDs that are not the tokenizer's padding ID.

  ## Options

    * `:compiler` - A module implementing `Nx.Defn.Compiler`. Defaults to `EXLA`.
    * `:batch_size` - The positive batch size to compile for. Inputs larger than this are processed
      in chunks. Defaults to 1 (lowest latency).
  """
  def compile_predictor(%BinClass.Classifier{} = classifier, opts \\ []) do
    BinClass.Predictor.compile(classifier, opts)
  end

  @doc """
  Compiles a predictor for one complete document.

  The returned function accepts exactly one string. It splits the entire input
  into overlapping token windows, predicts every chunk in batches, and returns
  each chunk's text, zero-based index, encoded token count, active-token count,
  original start/end byte offsets, and raw classification result.

  No text is normalized or truncated. `:start_byte` is inclusive and
  `:end_byte` is exclusive, and both refer to the exact string passed to the
  predictor. Empty or whitespace-only documents return no chunks and
  `status: :insufficient_input`.

  The document status is `:ok` when at least one chunk passes the classifier's
  `minimum_input_tokens` validation. Individual insufficient chunks remain in
  the output with their raw probabilities. A chunk's `:token_count` includes
  encoded padding special tokens present in the source; `:active_token_count`
  excludes them and is the value used for minimum-input validation. This
  function does not aggregate chunks or apply a document-level decision policy.

  ## Options

    * `:compiler` - A module implementing `Nx.Defn.Compiler`. Defaults to `EXLA`.
    * `:batch_size` - Positive maximum number of chunks predicted per model invocation.
      Defaults to `16`.
    * `:chunk_size` - Maximum token count per chunk. Defaults to the classifier's
      vector length and cannot exceed it.
    * `:chunk_overlap` - Overlap as a non-negative token count below
      `:chunk_size`, or as a fraction from `0.0` up to `1.0`. Defaults to `0.25`.

  ## Example

      predict =
        BinClass.compile_document_predictor(classifier,
          chunk_overlap: 128,
          batch_size: 32
        )

      %{status: :ok, chunks: chunks} = predict.(large_document)
  """
  def compile_document_predictor(%BinClass.Classifier{} = classifier, opts \\ []) do
    BinClass.DocumentPredictor.compile(classifier, opts)
  end

  defp validate_serving_options!(opts) do
    opts = Keyword.validate!(opts, compiler: EXLA, batch_size: 16, defn_options: [])

    opts
    |> Keyword.fetch!(:compiler)
    |> BinClass.Options.validate_compiler!()

    opts
    |> Keyword.fetch!(:batch_size)
    |> BinClass.Options.validate_positive_integer!(:batch_size)

    opts
    |> Keyword.fetch!(:defn_options)
    |> BinClass.Options.validate_keyword_list!(:defn_options)

    opts
  end
end
