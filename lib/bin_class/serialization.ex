defmodule BinClass.Serialization do
  @moduledoc false

  alias BinClass.{Classifier, DecisionPolicy, Model, Tokenizer}

  @doc false
  def serialize(%Classifier{} = classifier) do
    classifier = Classifier.validate!(classifier)

    model_params = Nx.backend_copy(classifier.model_params, Nx.BinaryBackend)

    %{
      tokenizer_json: Tokenizer.to_json(classifier.tokenizer),
      model_params: model_params,
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
      dropout_rate: classifier.dropout_rate,
      decision_policy: classifier.decision_policy,
      minimum_input_tokens: classifier.minimum_input_tokens,
      target_mode: classifier.target_mode,
      target_threshold: classifier.target_threshold
    }
    |> :erlang.term_to_binary()
  end

  @doc false
  def deserialize(binary) when is_binary(binary) do
    ensure_serialization_modules_loaded!()
    data = decode_data!(binary)

    architecture = fetch_field!(data, :architecture)
    model_revision = fetch_field!(data, :model_revision)

    tokenizer = data |> fetch_field!(:tokenizer_json) |> deserialize_tokenizer!()

    tokenizer_metadata = %{
      vocab_size: fetch_field!(data, :vocab_size),
      pad_token_id: fetch_field!(data, :pad_token_id),
      unk_token_id: fetch_field!(data, :unk_token_id)
    }

    vector_length = fetch_field!(data, :vector_length)

    %Classifier{
      tokenizer: tokenizer,
      model_params: fetch_field!(data, :model_params),
      vector_length: vector_length,
      vocab_size: tokenizer_metadata.vocab_size,
      pad_token_id: tokenizer_metadata.pad_token_id,
      unk_token_id: tokenizer_metadata.unk_token_id,
      labels: fetch_field!(data, :labels),
      accuracy: fetch_field!(data, :accuracy),
      epoch: fetch_field!(data, :epoch),
      architecture: architecture,
      model_revision: model_revision,
      learning_rate: fetch_field!(data, :learning_rate),
      dropout_rate: fetch_field!(data, :dropout_rate),
      decision_policy: fetch_field!(data, :decision_policy),
      minimum_input_tokens: fetch_field!(data, :minimum_input_tokens),
      target_mode: fetch_field!(data, :target_mode),
      target_threshold: fetch_field!(data, :target_threshold)
    }
    |> Classifier.validate!()
  end

  defp decode_data!(binary) do
    case :erlang.binary_to_term(binary, [:safe]) do
      data when is_map(data) ->
        data

      data ->
        raise ArgumentError, "Serialized classifier must contain a map, got: #{inspect(data)}"
    end
  end

  defp deserialize_tokenizer!(json) when is_binary(json) do
    case Tokenizers.Tokenizer.from_buffer(json) do
      {:ok, tokenizer} ->
        tokenizer

      {:error, reason} ->
        raise ArgumentError,
              "Serialized classifier contains invalid tokenizer JSON: #{inspect(reason)}"
    end
  end

  defp deserialize_tokenizer!(json) do
    raise ArgumentError,
          "Serialized classifier :tokenizer_json must be a binary, got: #{inspect(json)}"
  end

  defp fetch_field!(data, key) do
    case Map.fetch(data, key) do
      {:ok, value} -> value
      :error -> raise ArgumentError, "Serialized classifier is missing #{inspect(key)}"
    end
  end

  defp ensure_serialization_modules_loaded! do
    Enum.each(
      [
        Model,
        DecisionPolicy,
        Nx.Tensor,
        Nx.BinaryBackend,
        Nx.Type,
        Axon.ModelState
      ],
      &Code.ensure_loaded!/1
    )
  end
end
