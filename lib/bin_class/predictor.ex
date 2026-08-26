defmodule BinClass.Predictor do
  @moduledoc false

  alias BinClass.{DecisionPolicy, Labels, Model, Options, Serving, Vectorizer}

  @doc false
  def compile(%BinClass.Classifier{} = classifier, opts) do
    opts = Keyword.validate!(opts, compiler: EXLA, batch_size: 1)
    compiler = opts |> Keyword.fetch!(:compiler) |> Options.validate_compiler!()

    batch_size =
      opts
      |> Keyword.fetch!(:batch_size)
      |> Options.validate_positive_integer!(:batch_size)

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

    DecisionPolicy.validate!(decision_policy)
    Labels.validate!(labels)
    Model.validate_revision!(classifier.model_revision)
    Model.validate_vector_length!(architecture, vector_length)
    DecisionPolicy.validate_minimum_input_tokens!(minimum_input_tokens, vector_length)

    dropout_rate = validate_dropout_rate!(classifier.dropout_rate)

    model =
      Model.build(architecture, vocab_size,
        dropout_rate: dropout_rate,
        pad_token_id: pad_token_id
      )

    {_, predict_fn} = Axon.build(model, compiler: compiler)

    template = Nx.broadcast(pad_token_id, {batch_size, vector_length}) |> Nx.as_type(:u16)
    _ = predict_fn.(model_params, template)

    padding_vector = Vectorizer.build(tokenizer, "", vector_length, pad_token_id)

    predict_batch = fn texts ->
      {vectors, active_token_counts} =
        Enum.map(texts, fn text ->
          Vectorizer.build_with_active_token_count(
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
        Serving.decode_prediction(
          probs,
          labels,
          decision_policy,
          active_token_count,
          minimum_input_tokens
        )
      end)
    end

    fn input ->
      {texts, multi?} = Serving.validate_input(input)

      results =
        texts
        |> Stream.chunk_every(batch_size)
        |> Stream.flat_map(predict_batch)
        |> Enum.to_list()

      if multi?, do: results, else: List.first(results)
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
end
