defmodule BinClass.ServingTest do
  use ExUnit.Case
  alias BinClass.Serving

  test "Nx.Serving pipeline" do
    data = ["hello", "world"]
    tokenizer = BinClass.Tokenizer.train(data)
    vocab_size = BinClass.Tokenizer.vocab_size()

    # Init params
    model = BinClass.Model.build(1, vocab_size)
    template = Nx.broadcast(0, {1, 16}) |> Nx.as_type(:u16)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())

    serving =
      Serving.new(params, tokenizer, vector_length: 16, vocab_size: vocab_size, labels: [:a, :b])

    # Single prediction
    result = Nx.Serving.run(serving, "hello")
    assert %{label: _, confidence: _} = result

    # Batch prediction
    results = Nx.Serving.run(serving, ["hello", "world"])
    assert length(results) == 2
    assert match?([%{label: _, confidence: _}, %{label: _, confidence: _}], results)
  end

  test "Nx.Serving pipeline with map labels" do
    data = ["hello", "world"]
    tokenizer = BinClass.Tokenizer.train(data)
    vocab_size = BinClass.Tokenizer.vocab_size()

    # Init params
    model = BinClass.Model.build(1, vocab_size)
    template = Nx.broadcast(0, {1, 16}) |> Nx.as_type(:u16)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())

    labels = %{0 => :neg, 1 => :pos}

    serving =
      Serving.new(params, tokenizer, vector_length: 16, vocab_size: vocab_size, labels: labels)

    # Single prediction
    result = Nx.Serving.run(serving, "hello")
    assert result.label in [:neg, :pos]
    assert Map.has_key?(result.probabilities, :neg)
    assert Map.has_key?(result.probabilities, :pos)
  end
end
