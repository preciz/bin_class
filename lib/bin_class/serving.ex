defmodule BinClass.Serving do
  @moduledoc """
  Defines the Nx.Serving pipeline for binary classification.
  """

  alias BinClass.{Model, Vectorizer, Tokenizer}

  def new(model_params, tokenizer, opts \\ []) do
    vector_length = Keyword.get(opts, :vector_length, 512)
    vocab_size = Keyword.get(opts, :vocab_size, Tokenizer.vocab_size())
    labels = Keyword.get(opts, :labels, [0, 1])
    batch_size = Keyword.get(opts, :batch_size, 16)
    compiler = Keyword.get(opts, :compiler, EXLA)
    defn_options = Keyword.get(opts, :defn_options, [])
    model_version = Keyword.get(opts, :model_version, 1)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)
    policy = Keyword.get(opts, :decision_policy) || decision_policy(model_version)
    positive_threshold = Keyword.get(opts, :positive_threshold, policy.positive_threshold)
    min_positive_tokens = Keyword.get(opts, :min_positive_tokens, policy.min_positive_tokens)

    use_decision_policy? =
      decision_policy?(model_version) or Keyword.has_key?(opts, :decision_policy) or
        Keyword.has_key?(opts, :positive_threshold) or
        Keyword.has_key?(opts, :min_positive_tokens)

    model = Model.build(model_version, vocab_size, dropout_rate: dropout_rate)
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

      vectors =
        texts
        |> Enum.map(&Vectorizer.build(tokenizer, &1, vector_length))

      active_lengths =
        Enum.map(vectors, fn vector ->
          Enum.count(vector, &(&1 != 0))
        end)

      batch =
        vectors
        |> Enum.map(&Nx.tensor(&1, type: :u16))
        |> Nx.Batch.stack()

      {batch, {multi?, active_lengths}}
    end)
    |> Nx.Serving.client_postprocessing(fn {batch_output, _server_info},
                                           {multi?, active_lengths} ->
      # batch_output is [batch_size, 2]
      results =
        batch_output
        |> Nx.to_list()
        |> Enum.zip(active_lengths)
        |> Enum.map(fn {probs, active_length} ->
          if use_decision_policy? do
            decode_prediction(
              probs,
              labels,
              positive_threshold,
              active_length,
              min_positive_tokens
            )
          else
            decode_prediction(probs, labels)
          end
        end)

      if multi?, do: results, else: List.first(results)
    end)
  end

  @doc false
  def validate_input(input) when is_list(input), do: {input, true}
  def validate_input(input), do: {[input], false}

  @doc false
  def decode_prediction(probs, labels) do
    max_prob = Enum.max(probs)
    max_index = Enum.find_index(probs, &(&1 == max_prob))
    build_prediction(probs, labels, max_index, max_prob)
  end

  @doc false
  def decode_prediction(probs, labels, positive_threshold, active_length, min_positive_tokens) do
    positive_prob = Enum.at(probs, 1)

    max_index =
      if positive_prob >= positive_threshold and active_length >= min_positive_tokens do
        1
      else
        0
      end

    build_prediction(probs, labels, max_index, Enum.at(probs, max_index))
  end

  defp build_prediction(probs, labels, max_index, max_prob) do
    label =
      if is_map(labels) do
        Map.get(labels, max_index)
      else
        Enum.at(labels, max_index)
      end

    probabilities =
      if is_map(labels) do
        probs
        |> Enum.with_index()
        |> Map.new(fn {prob, idx} -> {Map.get(labels, idx), prob} end)
      else
        Enum.zip(labels, probs) |> Map.new()
      end

    %{
      label: label,
      confidence: max_prob,
      probabilities: probabilities
    }
  end

  @doc false
  def decision_policy(model_version) do
    %{
      positive_threshold: positive_threshold(model_version),
      min_positive_tokens: min_positive_tokens(model_version)
    }
  end

  @doc false
  def decision_policy?(model_version), do: model_version in [7, :conservative_cnn]

  defp positive_threshold(model_version) when model_version in [7, :conservative_cnn], do: 0.6
  defp positive_threshold(_model_version), do: 0.5

  defp min_positive_tokens(model_version) when model_version in [7, :conservative_cnn], do: 64
  defp min_positive_tokens(_model_version), do: 0
end
