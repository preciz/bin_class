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

  test "Serving.new with custom options" do
    data = ["hello"]
    tokenizer = BinClass.Tokenizer.train(data)
    vocab_size = BinClass.Tokenizer.vocab_size()
    model = BinClass.Model.build(1, vocab_size)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(Nx.broadcast(0, {1, 16}) |> Nx.as_type(:u16), Axon.ModelState.empty())

    serving =
      Serving.new(params, tokenizer,
        batch_size: 4,
        compiler: EXLA,
        defn_options: [compiler: EXLA]
      )

    assert %Nx.Serving{} = serving
  end

  test "Serving.new with no options" do
    data = ["hello"]
    tokenizer = BinClass.Tokenizer.train(data)
    vocab_size = BinClass.Tokenizer.vocab_size()
    model = BinClass.Model.build(1, vocab_size)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(Nx.broadcast(0, {1, 16}) |> Nx.as_type(:u16), Axon.ModelState.empty())

    serving = Serving.new(params, tokenizer)
    assert %Nx.Serving{} = serving
  end

  test "v7 decision policy is conservative for uncertain and low-signal positives" do
    labels = %{0 => :negative, 1 => :positive}
    policy = Serving.decision_policy(7)

    refute Serving.decision_policy?(1)
    assert Serving.decision_policy?(7)
    assert policy.positive_threshold == 0.6
    assert policy.min_positive_tokens == 64

    assert %{label: :negative} =
             Serving.decode_prediction([0.45, 0.55], labels, 0.6, 128, 64)

    assert %{label: :negative} =
             Serving.decode_prediction([0.1, 0.9], labels, 0.6, 32, 64)

    assert %{label: :positive} =
             Serving.decode_prediction([0.35, 0.65], labels, 0.6, 128, 64)

    assert %{label: :positive} = Serving.decode_prediction([0.35, 0.65], labels)
  end
end
