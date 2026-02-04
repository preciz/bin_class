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

  """

  @doc """
  Saves the classifier to a file.
  """
  def save(%BinClass.Classifier{} = classifier, path) do
    # Ensure model params are on Binary backend for safe serialization
    model_params_binary = Nx.backend_copy(classifier.model_params, Nx.BinaryBackend)

    data = %{
      tokenizer_json: BinClass.Tokenizer.to_json(classifier.tokenizer),
      model_params: model_params_binary,
      vector_length: classifier.vector_length,
      vocab_size: classifier.vocab_size,
      labels: classifier.labels,
      accuracy: classifier.accuracy,
      epoch: classifier.epoch
    }

    binary = :erlang.term_to_binary(data)
    File.write!(path, binary)
  end

  @doc """
  Loads a saved model from a file and returns an Nx.Serving struct.
  """
  def load(path, opts \\ []) do
    binary = File.read!(path)
    data = :erlang.binary_to_term(binary)

    {:ok, tokenizer} = Tokenizers.Tokenizer.from_buffer(data.tokenizer_json)

    serving_opts =
      Keyword.merge(opts,
        vector_length: data.vector_length,
        vocab_size: data.vocab_size,
        labels: data.labels
      )

    BinClass.Serving.new(data.model_params, tokenizer, serving_opts)
  end
end
