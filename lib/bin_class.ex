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
      classifier = BinClass.Trainer.train(data,
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
    BinClass.Model.validate_name!(classifier.architecture)
    BinClass.Model.validate_revision!(classifier.model_revision)
    BinClass.Model.validate_vector_length!(classifier.architecture, classifier.vector_length)
    BinClass.Labels.validate!(classifier.labels)
    BinClass.DecisionPolicy.validate!(classifier.decision_policy)
    dropout_rate = validate_dropout_rate!(classifier.dropout_rate)
    target_mode = validate_target_mode!(classifier.target_mode)
    target_threshold = validate_target_threshold!(classifier.target_threshold)

    BinClass.DecisionPolicy.validate_minimum_input_tokens!(
      classifier.minimum_input_tokens,
      classifier.vector_length
    )

    BinClass.Tokenizer.validate_metadata!(classifier.tokenizer, %{
      vocab_size: classifier.vocab_size,
      pad_token_id: classifier.pad_token_id,
      unk_token_id: classifier.unk_token_id
    })

    # Ensure model params are on Binary backend for safe serialization
    model_params_binary = Nx.backend_copy(classifier.model_params, Nx.BinaryBackend)

    data = %{
      tokenizer_json: BinClass.Tokenizer.to_json(classifier.tokenizer),
      model_params: model_params_binary,
      vector_length: classifier.vector_length,
      vocab_size: classifier.vocab_size,
      pad_token_id: classifier.pad_token_id,
      unk_token_id: classifier.unk_token_id,
      labels: classifier.labels,
      accuracy: classifier.accuracy,
      epoch: classifier.epoch,
      architecture: classifier.architecture,
      model_revision: classifier.model_revision,
      learning_rate: classifier.learning_rate,
      dropout_rate: dropout_rate,
      decision_policy: classifier.decision_policy,
      minimum_input_tokens: classifier.minimum_input_tokens,
      target_mode: target_mode,
      target_threshold: target_threshold
    }

    :erlang.term_to_binary(data)
  end

  @doc """
  Loads a saved model from a file and returns an Nx.Serving struct.
  """
  def load(path, opts \\ []) do
    opts = validate_serving_options!(opts)

    path
    |> load_classifier()
    |> serving(opts)
  end

  @doc """
  Deserializes a saved model from a binary and returns an Nx.Serving struct.
  """
  def deserialize(binary, opts \\ []) when is_binary(binary) do
    opts = validate_serving_options!(opts)

    binary
    |> deserialize_classifier()
    |> serving(opts)
  end

  @doc """
  Builds an `Nx.Serving` from a classifier.

  Model architecture, tokenizer metadata, labels, vector length, decision
  policy, and minimum-input validation always come from the classifier.

  ## Options

    * `:compiler` - The Nx compiler to use. Defaults to `EXLA`.
    * `:batch_size` - The serving batch size. Defaults to `16`.
    * `:defn_options` - Options passed to the compiled definition. Defaults to `[]`.
  """
  def serving(%BinClass.Classifier{} = classifier, opts \\ []) do
    opts = validate_serving_options!(opts)
    dropout_rate = validate_dropout_rate!(classifier.dropout_rate)

    serving_opts =
      Keyword.merge(opts,
        vector_length: classifier.vector_length,
        labels: classifier.labels,
        architecture: classifier.architecture,
        dropout_rate: dropout_rate,
        decision_policy: classifier.decision_policy,
        minimum_input_tokens: classifier.minimum_input_tokens
      )

    BinClass.Model.validate_revision!(classifier.model_revision)

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
    ensure_serialization_modules_loaded!()
    data = :erlang.binary_to_term(binary, [:safe])

    architecture =
      data
      |> fetch_serialized_field!(:architecture)
      |> BinClass.Model.validate_name!()

    model_revision =
      data
      |> fetch_serialized_field!(:model_revision)
      |> BinClass.Model.validate_revision!()

    {:ok, tokenizer} =
      data
      |> fetch_serialized_field!(:tokenizer_json)
      |> Tokenizers.Tokenizer.from_buffer()

    tokenizer_metadata = %{
      vocab_size: fetch_serialized_field!(data, :vocab_size),
      pad_token_id: fetch_serialized_field!(data, :pad_token_id),
      unk_token_id: fetch_serialized_field!(data, :unk_token_id)
    }

    BinClass.Tokenizer.validate_metadata!(tokenizer, tokenizer_metadata)

    vector_length = fetch_serialized_field!(data, :vector_length)
    BinClass.Model.validate_vector_length!(architecture, vector_length)

    dropout_rate =
      data
      |> fetch_serialized_field!(:dropout_rate)
      |> validate_dropout_rate!()

    target_mode =
      data
      |> fetch_serialized_field!(:target_mode)
      |> validate_target_mode!()

    target_threshold =
      data
      |> fetch_serialized_field!(:target_threshold)
      |> validate_target_threshold!()

    %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: fetch_serialized_field!(data, :model_params),
      vector_length: vector_length,
      vocab_size: tokenizer_metadata.vocab_size,
      pad_token_id: tokenizer_metadata.pad_token_id,
      unk_token_id: tokenizer_metadata.unk_token_id,
      labels:
        data
        |> fetch_serialized_field!(:labels)
        |> BinClass.Labels.validate!(),
      accuracy: fetch_serialized_field!(data, :accuracy),
      epoch: fetch_serialized_field!(data, :epoch),
      architecture: architecture,
      model_revision: model_revision,
      learning_rate: fetch_serialized_field!(data, :learning_rate),
      dropout_rate: dropout_rate,
      decision_policy:
        data
        |> fetch_serialized_field!(:decision_policy)
        |> BinClass.DecisionPolicy.validate!(),
      minimum_input_tokens:
        data
        |> fetch_serialized_field!(:minimum_input_tokens)
        |> BinClass.DecisionPolicy.validate_minimum_input_tokens!(vector_length),
      target_mode: target_mode,
      target_threshold: target_threshold
    }
  end

  @doc """
  Compiles the classifier into a highly optimized, in-process prediction function.

  This is intended for scenarios where lowest possible latency is required
  and batching (provided by Nx.Serving) is not necessary (e.g. CLI tools,
  single-user scripts, or very low-concurrency high-speed inference).

  Returns an anonymous function that takes a text (string) or list of texts
  and returns the classification results. Inputs below the classifier's
  `minimum_input_tokens` return `label: :insufficient_input` and
  `confidence: nil`, while retaining raw class probabilities. Every result
  includes `:active_token_count`, defined consistently as the number of
  truncated token IDs that are not the tokenizer's padding ID.

  ## Options

    * `:compiler` - The compiler to use. Defaults to `EXLA`.
    * `:batch_size` - The batch size to compile for. Inputs larger than this are processed
      in chunks. Defaults to 1 (lowest latency).
  """
  def compile_predictor(%BinClass.Classifier{} = classifier, opts \\ []) do
    opts = Keyword.validate!(opts, compiler: EXLA, batch_size: 1)
    compiler = Keyword.fetch!(opts, :compiler)
    batch_size = Keyword.fetch!(opts, :batch_size)

    unless is_integer(batch_size) and batch_size > 0 do
      raise ArgumentError, ":batch_size must be a positive integer, got: #{inspect(batch_size)}"
    end

    %{
      architecture: architecture,
      vocab_size: vocab_size,
      pad_token_id: pad_token_id,
      tokenizer: tokenizer,
      vector_length: vector_length,
      model_params: model_params,
      labels: labels,
      decision_policy: decision_policy,
      minimum_input_tokens: minimum_input_tokens
    } = classifier

    BinClass.DecisionPolicy.validate!(decision_policy)
    BinClass.Labels.validate!(labels)
    BinClass.Model.validate_revision!(classifier.model_revision)
    BinClass.Model.validate_vector_length!(architecture, vector_length)

    BinClass.DecisionPolicy.validate_minimum_input_tokens!(
      minimum_input_tokens,
      vector_length
    )

    dropout_rate = validate_dropout_rate!(classifier.dropout_rate)

    model =
      BinClass.Model.build(architecture, vocab_size,
        dropout_rate: dropout_rate,
        pad_token_id: pad_token_id
      )

    {_, predict_fn} = Axon.build(model, compiler: compiler)

    # Compile the prediction function specifically for the given batch size
    template = Nx.broadcast(pad_token_id, {batch_size, vector_length}) |> Nx.as_type(:u16)

    # Warmup / JIT compilation
    _ = predict_fn.(model_params, template)

    padding_vector =
      BinClass.Vectorizer.build(tokenizer, "", vector_length, pad_token_id)

    predict_batch = fn texts ->
      {vectors, active_token_counts} =
        Enum.map(texts, fn text ->
          BinClass.Vectorizer.build_with_active_token_count(
            tokenizer,
            text,
            vector_length,
            pad_token_id
          )
        end)
        |> Enum.unzip()

      batch_tensor =
        vectors
        |> Kernel.++(List.duplicate(padding_vector, batch_size - length(vectors)))
        |> Nx.tensor(type: :u16)

      predict_fn.(model_params, batch_tensor)
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_list()
      |> Enum.take(length(texts))
      |> Enum.zip(active_token_counts)
      |> Enum.map(fn {probs, active_token_count} ->
        BinClass.Serving.decode_prediction(
          probs,
          labels,
          decision_policy,
          active_token_count,
          minimum_input_tokens
        )
      end)
    end

    fn input ->
      {texts, multi?} = BinClass.Serving.validate_input(input)

      results =
        texts
        |> Stream.chunk_every(batch_size)
        |> Stream.flat_map(predict_batch)
        |> Enum.to_list()

      if multi?, do: results, else: List.first(results)
    end
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

    * `:compiler` - The Nx compiler to use. Defaults to `EXLA`.
    * `:batch_size` - Maximum number of chunks predicted per model invocation.
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

  defp fetch_serialized_field!(data, key) do
    case Map.fetch(data, key) do
      {:ok, value} -> value
      :error -> raise ArgumentError, "Serialized classifier is missing #{inspect(key)}"
    end
  end

  defp validate_dropout_rate!(dropout_rate)
       when is_number(dropout_rate) and dropout_rate >= 0 and dropout_rate < 1,
       do: dropout_rate

  defp validate_dropout_rate!(dropout_rate) do
    raise ArgumentError,
          "Classifier :dropout_rate must be a number greater than or equal to 0 and below 1, " <>
            "got: #{inspect(dropout_rate)}"
  end

  defp validate_target_mode!(target_mode) when target_mode in [:hard, :soft], do: target_mode

  defp validate_target_mode!(target_mode) do
    raise ArgumentError,
          "Classifier :target_mode must be :hard or :soft, got: #{inspect(target_mode)}"
  end

  defp validate_target_threshold!(target_threshold)
       when is_number(target_threshold) and target_threshold >= 0 and target_threshold <= 1,
       do: target_threshold

  defp validate_target_threshold!(target_threshold) do
    raise ArgumentError,
          "Classifier :target_threshold must be a number between 0 and 1, " <>
            "got: #{inspect(target_threshold)}"
  end

  defp validate_serving_options!(opts) do
    Keyword.validate!(opts, [
      :compiler,
      :batch_size,
      :defn_options
    ])
  end

  defp ensure_serialization_modules_loaded! do
    Enum.each(
      [
        BinClass.Model,
        BinClass.DecisionPolicy,
        BinClass.Trainer,
        Nx.Tensor,
        Nx.BinaryBackend,
        Nx.Type,
        Axon.ModelState
      ],
      &Code.ensure_loaded!/1
    )
  end
end
