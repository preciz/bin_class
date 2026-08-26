defmodule BinClass.Serving do
  @moduledoc false

  alias BinClass.{DecisionPolicy, Model, Tokenizer, Vectorizer}

  @doc false
  def new(model_params, tokenizer, opts) do
    opts =
      Keyword.validate!(opts, [
        :vector_length,
        :labels,
        :batch_size,
        :compiler,
        :defn_options,
        :architecture,
        :dropout_rate,
        :decision_policy,
        :minimum_input_tokens
      ])

    vector_length = Keyword.fetch!(opts, :vector_length)
    architecture = Keyword.fetch!(opts, :architecture)
    batch_size = Keyword.fetch!(opts, :batch_size)
    compiler = Keyword.fetch!(opts, :compiler)
    defn_options = Keyword.fetch!(opts, :defn_options)
    dropout_rate = Keyword.fetch!(opts, :dropout_rate)

    tokenizer_metadata = Tokenizer.metadata(tokenizer)
    vocab_size = tokenizer_metadata.vocab_size
    pad_token_id = tokenizer_metadata.pad_token_id
    labels = Keyword.fetch!(opts, :labels)
    policy = Keyword.fetch!(opts, :decision_policy)
    minimum_input_tokens = Keyword.fetch!(opts, :minimum_input_tokens)

    model =
      Model.build(architecture, vocab_size,
        dropout_rate: dropout_rate,
        pad_token_id: pad_token_id
      )

    {_, predict_fn} = Axon.build(model, compiler: compiler)

    Nx.Serving.new(
      fn _defn_options ->
        fn inputs ->
          # Inputs are already batched by client_preprocessing
          predict_fn.(model_params, inputs)
          |> Nx.backend_transfer(Nx.BinaryBackend)
        end
      end,
      compiler: compiler,
      defn_options: defn_options
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.client_preprocessing(fn input ->
      {texts, multi?} = validate_input(input)

      {vectors, active_token_counts} =
        texts
        |> Enum.map(
          &Vectorizer.build_with_active_token_count(
            tokenizer,
            &1,
            vector_length,
            pad_token_id
          )
        )
        |> Enum.unzip()

      batch =
        vectors
        |> Enum.map(&Nx.tensor(&1, type: :u16))
        |> Nx.Batch.stack()

      {batch, {multi?, active_token_counts}}
    end)
    |> Nx.Serving.client_postprocessing(fn {batch_output, _server_info},
                                           {multi?, active_token_counts} ->
      # batch_output is [batch_size, 2]
      results =
        batch_output
        |> Nx.to_list()
        |> Enum.zip(active_token_counts)
        |> Enum.map(fn {probs, active_token_count} ->
          decode_prediction(probs, labels, policy, active_token_count, minimum_input_tokens)
        end)

      if multi?, do: results, else: List.first(results)
    end)
  end

  @doc false
  def validate_input(input) when is_binary(input), do: {[input], false}

  def validate_input([_text | _texts] = input) do
    if Enum.all?(input, &is_binary/1) do
      {input, true}
    else
      invalid_input!(input)
    end
  end

  def validate_input(input), do: invalid_input!(input)

  @doc false
  def decode_prediction(probs, labels, policy, active_token_count, minimum_input_tokens) do
    prediction =
      if active_token_count < minimum_input_tokens do
        %{
          label: :insufficient_input,
          confidence: nil,
          probabilities: build_probabilities(probs, labels)
        }
      else
        max_index = DecisionPolicy.class_index(probs, policy)
        build_prediction(probs, labels, max_index, Enum.at(probs, max_index))
      end

    Map.put(prediction, :active_token_count, active_token_count)
  end

  defp build_prediction(probs, labels, max_index, max_prob) do
    label =
      if is_map(labels) do
        Map.get(labels, max_index)
      else
        Enum.at(labels, max_index)
      end

    %{
      label: label,
      confidence: max_prob,
      probabilities: build_probabilities(probs, labels)
    }
  end

  defp build_probabilities(probs, labels) when is_map(labels) do
    probs
    |> Enum.with_index()
    |> Map.new(fn {prob, idx} -> {Map.get(labels, idx), prob} end)
  end

  defp build_probabilities(probs, labels), do: Enum.zip(labels, probs) |> Map.new()

  defp invalid_input!(input) do
    raise ArgumentError,
          "prediction input must be a string or non-empty list of strings, got: " <>
            inspect(input)
  end
end
