defmodule BinClass.ServingTest do
  use ExUnit.Case
  alias BinClass.Serving

  test "Nx.Serving pipeline" do
    data = ["hello", "world"]
    tokenizer = BinClass.Tokenizer.train(data)
    vocab_size = BinClass.Tokenizer.vocab_size()

    # Init params
    model = BinClass.Model.build(vocab_size)
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
end
