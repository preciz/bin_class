defmodule BinClass.Serialization do
  @moduledoc false

  alias BinClass.{Classifier, DecisionPolicy, Labels, Model, Tokenizer}

  @doc false
  def serialize(%Classifier{} = classifier) do
    Model.validate_name!(classifier.architecture)
    Model.validate_revision!(classifier.model_revision)
    Model.validate_vector_length!(classifier.architecture, classifier.vector_length)
    Labels.validate!(classifier.labels)
    DecisionPolicy.validate!(classifier.decision_policy)
    dropout_rate = validate_dropout_rate!(classifier.dropout_rate)
    target_mode = validate_target_mode!(classifier.target_mode)
    target_threshold = validate_target_threshold!(classifier.target_threshold)

    DecisionPolicy.validate_minimum_input_tokens!(
      classifier.minimum_input_tokens,
      classifier.vector_length
    )

    Tokenizer.validate_metadata!(classifier.tokenizer, %{
      vocab_size: classifier.vocab_size,
      pad_token_id: classifier.pad_token_id,
      unk_token_id: classifier.unk_token_id
    })

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
      dropout_rate: dropout_rate,
      decision_policy: classifier.decision_policy,
      minimum_input_tokens: classifier.minimum_input_tokens,
      target_mode: target_mode,
      target_threshold: target_threshold
    }
    |> :erlang.term_to_binary()
  end

  @doc false
  def deserialize(binary) when is_binary(binary) do
    ensure_serialization_modules_loaded!()
    data = :erlang.binary_to_term(binary, [:safe])

    architecture =
      data
      |> fetch_field!(:architecture)
      |> Model.validate_name!()

    model_revision =
      data
      |> fetch_field!(:model_revision)
      |> Model.validate_revision!()

    {:ok, tokenizer} =
      data
      |> fetch_field!(:tokenizer_json)
      |> Tokenizers.Tokenizer.from_buffer()

    tokenizer_metadata = %{
      vocab_size: fetch_field!(data, :vocab_size),
      pad_token_id: fetch_field!(data, :pad_token_id),
      unk_token_id: fetch_field!(data, :unk_token_id)
    }

    Tokenizer.validate_metadata!(tokenizer, tokenizer_metadata)

    vector_length = fetch_field!(data, :vector_length)
    Model.validate_vector_length!(architecture, vector_length)

    dropout_rate =
      data
      |> fetch_field!(:dropout_rate)
      |> validate_dropout_rate!()

    target_mode =
      data
      |> fetch_field!(:target_mode)
      |> validate_target_mode!()

    target_threshold =
      data
      |> fetch_field!(:target_threshold)
      |> validate_target_threshold!()

    %Classifier{
      tokenizer: tokenizer,
      model_params: fetch_field!(data, :model_params),
      vector_length: vector_length,
      vocab_size: tokenizer_metadata.vocab_size,
      pad_token_id: tokenizer_metadata.pad_token_id,
      unk_token_id: tokenizer_metadata.unk_token_id,
      labels: data |> fetch_field!(:labels) |> Labels.validate!(),
      accuracy: fetch_field!(data, :accuracy),
      epoch: fetch_field!(data, :epoch),
      architecture: architecture,
      model_revision: model_revision,
      learning_rate: fetch_field!(data, :learning_rate),
      dropout_rate: dropout_rate,
      decision_policy:
        data
        |> fetch_field!(:decision_policy)
        |> DecisionPolicy.validate!(),
      minimum_input_tokens:
        data
        |> fetch_field!(:minimum_input_tokens)
        |> DecisionPolicy.validate_minimum_input_tokens!(vector_length),
      target_mode: target_mode,
      target_threshold: target_threshold
    }
  end

  defp fetch_field!(data, key) do
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

  defp ensure_serialization_modules_loaded! do
    Enum.each(
      [
        Model,
        DecisionPolicy,
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
