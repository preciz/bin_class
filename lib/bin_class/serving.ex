defmodule BinClass.Serving do
  @moduledoc """
  Defines the Nx.Serving pipeline for binary classification.
  """

  alias BinClass.{Model, Vectorizer, Tokenizer}

  def new(model_params, tokenizer, opts \\ []) do
    vector_length = Keyword.get(opts, :vector_length, 256)
    vocab_size = Keyword.get(opts, :vocab_size, Tokenizer.vocab_size())
    labels = Keyword.get(opts, :labels, [0, 1])
    batch_size = Keyword.get(opts, :batch_size, 16)
    compiler = Keyword.get(opts, :compiler, EXLA)
    defn_options = Keyword.get(opts, :defn_options, [])

    model = Model.build(vocab_size)
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

      batch =
        texts
        |> Enum.map(fn text ->
          Vectorizer.build(tokenizer, text, vector_length)
          |> Nx.tensor(type: :u16)
        end)
        |> Nx.Batch.stack()

      {batch, multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {batch_output, _server_info}, multi? ->
      # batch_output is [batch_size, 2]
      results =
        batch_output
        |> Nx.to_list()
        |> Enum.map(fn probs -> decode_prediction(probs, labels) end)

      if multi?, do: results, else: List.first(results)
    end)
  end

  defp validate_input(input) when is_list(input), do: {input, true}
  defp validate_input(input), do: {[input], false}

  defp decode_prediction(probs, labels) do
    max_prob = Enum.max(probs)
    max_index = Enum.find_index(probs, &(&1 == max_prob))

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
end
