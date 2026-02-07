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
        labels: %{0 => :negative, 1 => :positive}
      )

      # Save the model
      BinClass.save(classifier, "model.bin")

  ### Prediction

      # Load the model as a serving (supports custom compiler/defn_options)
      serving = BinClass.load("model.bin", compiler: EXLA)

      result = Nx.Serving.run(serving, "Some text to classify")
      # result is %{label: :positive, confidence: 0.99, ...}

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
  """
  def serialize(%BinClass.Classifier{} = classifier) do
    # Ensure model params are on Binary backend for safe serialization
    model_params_binary = Nx.backend_copy(classifier.model_params, Nx.BinaryBackend)

    data = %{
      tokenizer_json: BinClass.Tokenizer.to_json(classifier.tokenizer),
      model_params: model_params_binary,
      vector_length: classifier.vector_length,
      vocab_size: classifier.vocab_size,
      labels: classifier.labels,
      accuracy: classifier.accuracy,
      epoch: classifier.epoch,
      model_version: classifier.model_version,
      learning_rate: classifier.learning_rate,
      dropout_rate: classifier.dropout_rate
    }

    :erlang.term_to_binary(data)
  end

  @doc """
  Loads a saved model from a file and returns an Nx.Serving struct.
  """
  def load(path, opts \\ []) do
    classifier = load_classifier(path)

    serving_opts =
      Keyword.merge(opts,
        vector_length: classifier.vector_length,
        vocab_size: classifier.vocab_size,
        labels: classifier.labels,
        model_version: classifier.model_version,
        dropout_rate: classifier.dropout_rate || 0.2
      )

    BinClass.Serving.new(classifier.model_params, classifier.tokenizer, serving_opts)
  end

  @doc """
  Deserializes a saved model from a binary and returns an Nx.Serving struct.
  """
  def deserialize(binary, opts \\ []) when is_binary(binary) do
    classifier = deserialize_classifier(binary)

    serving_opts =
      Keyword.merge(opts,
        vector_length: classifier.vector_length,
        vocab_size: classifier.vocab_size,
        labels: classifier.labels,
        model_version: classifier.model_version,
        dropout_rate: classifier.dropout_rate || 0.2
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
  """
  def deserialize_classifier(binary) when is_binary(binary) do
    data = :erlang.binary_to_term(binary)

    {:ok, tokenizer} = Tokenizers.Tokenizer.from_buffer(data.tokenizer_json)

    %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: data.model_params,
      vector_length: data.vector_length,
      vocab_size: data.vocab_size,
      labels: data.labels,
      accuracy: Map.get(data, :accuracy),
      epoch: Map.get(data, :epoch),
      model_version: Map.get(data, :model_version, 1),
      learning_rate: Map.get(data, :learning_rate),
      dropout_rate: Map.get(data, :dropout_rate, 0.2)
    }
  end

  @doc """
  Compiles the classifier into a highly optimized, in-process prediction function.

  This is intended for scenarios where lowest possible latency is required
  and batching (provided by Nx.Serving) is not necessary (e.g. CLI tools,
  single-user scripts, or very low-concurrency high-speed inference).

  Returns an anonymous function that takes a text (string) or list of texts
  and returns the classification results.

  ## Options

    * `:compiler` - The compiler to use. Defaults to `EXLA`.
    * `:batch_size` - The batch size to compile for. Defaults to 1 (lowest latency).
  """
  def compile_predictor(%BinClass.Classifier{} = classifier, opts \\ []) do
    compiler = Keyword.get(opts, :compiler, EXLA)
    batch_size = Keyword.get(opts, :batch_size, 1)

    model =
      BinClass.Model.build(classifier.model_version, classifier.vocab_size,
        dropout_rate: classifier.dropout_rate || 0.2
      )

    {_, predict_fn} = Axon.build(model, compiler: compiler)

    # Compile the prediction function specifically for the given batch size
    template = Nx.broadcast(0, {batch_size, classifier.vector_length}) |> Nx.as_type(:u16)

    # Warmup / JIT compilation
    _ = predict_fn.(classifier.model_params, template)

    fn input ->
      {texts, multi?} = BinClass.Serving.validate_input(input)

      # Pad or truncate to batch_size
      batch_texts =
        if length(texts) < batch_size do
          # Pad with empty strings if smaller than compiled batch size
          texts ++ List.duplicate("", batch_size - length(texts))
        else
          # Take only up to compiled batch size
          Enum.take(texts, batch_size)
        end

      batch_tensor =
        batch_texts
        |> Enum.map(fn text ->
          BinClass.Vectorizer.build(classifier.tokenizer, text, classifier.vector_length)
          |> Nx.tensor(type: :u16)
        end)
        |> Nx.Batch.stack()

      # Run inference directly
      batch_output =
        predict_fn.(classifier.model_params, batch_tensor)
        |> Nx.backend_transfer(Nx.BinaryBackend)

      # Decode only the requested number of items
      results =
        batch_output
        |> Nx.to_list()
        |> Enum.take(length(texts))
        |> Enum.map(fn probs -> BinClass.Serving.decode_prediction(probs, classifier.labels) end)

      if multi?, do: results, else: List.first(results)
    end
  end
end
