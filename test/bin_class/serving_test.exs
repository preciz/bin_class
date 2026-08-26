defmodule BinClass.ServingTest do
  use ExUnit.Case
  alias BinClass.Serving

  test "Nx.Serving pipeline" do
    data = ["hello", "world"]
    tokenizer = BinClass.Tokenizer.train(data)
    metadata = BinClass.Tokenizer.metadata(tokenizer)

    # Init params
    model =
      BinClass.Model.build(:cnn, metadata.vocab_size, pad_token_id: metadata.pad_token_id)

    template = Nx.broadcast(0, {1, 16}) |> Nx.as_type(:u16)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())

    serving =
      Serving.new(params, tokenizer,
        architecture: :cnn,
        vector_length: 16,
        labels: ["a", "b"]
      )

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
    metadata = BinClass.Tokenizer.metadata(tokenizer)

    # Init params
    model =
      BinClass.Model.build(:cnn, metadata.vocab_size, pad_token_id: metadata.pad_token_id)

    template = Nx.broadcast(0, {1, 16}) |> Nx.as_type(:u16)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())

    labels = %{0 => "negative", 1 => "positive"}

    serving =
      Serving.new(params, tokenizer,
        architecture: :cnn,
        vector_length: 16,
        labels: labels
      )

    # Single prediction
    result = Nx.Serving.run(serving, "hello")
    assert result.label in ["negative", "positive"]
    assert Map.has_key?(result.probabilities, "negative")
    assert Map.has_key?(result.probabilities, "positive")
  end

  test "Serving.new with custom options" do
    data = ["hello"]
    tokenizer = BinClass.Tokenizer.train(data)
    metadata = BinClass.Tokenizer.metadata(tokenizer)

    model =
      BinClass.Model.build(:cnn, metadata.vocab_size, pad_token_id: metadata.pad_token_id)

    {init_fn, _} = Axon.build(model)
    params = init_fn.(Nx.broadcast(0, {1, 16}) |> Nx.as_type(:u16), Axon.ModelState.empty())

    serving =
      Serving.new(params, tokenizer,
        architecture: :cnn,
        batch_size: 4,
        compiler: EXLA,
        defn_options: [compiler: EXLA]
      )

    assert %Nx.Serving{} = serving
  end

  test "Serving.new with no options" do
    data = ["hello"]
    tokenizer = BinClass.Tokenizer.train(data)
    metadata = BinClass.Tokenizer.metadata(tokenizer)

    model =
      BinClass.Model.build(:cnn, metadata.vocab_size, pad_token_id: metadata.pad_token_id)

    {init_fn, _} = Axon.build(model)
    params = init_fn.(Nx.broadcast(0, {1, 16}) |> Nx.as_type(:u16), Axon.ModelState.empty())

    serving = Serving.new(params, tokenizer)
    assert %Nx.Serving{} = serving
  end

  test "unknown tokens count as active tokens rather than padding" do
    tokenizer = BinClass.Tokenizer.train(["known"])
    metadata = BinClass.Tokenizer.metadata(tokenizer)
    text = List.duplicate("☃", 64) |> Enum.join(" ")

    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)
    raw_ids = Tokenizers.Encoding.get_ids(encoding)
    assert length(raw_ids) == 64
    assert Enum.all?(raw_ids, &(&1 == metadata.unk_token_id))

    model =
      BinClass.Model.build(:cnn, metadata.vocab_size, pad_token_id: metadata.pad_token_id)

    {init_fn, _} = Axon.build(model)
    template = Nx.broadcast(metadata.pad_token_id, {1, 64}) |> Nx.as_type(:u16)
    params = init_fn.(template, Axon.ModelState.empty())

    serving =
      Serving.new(params, tokenizer,
        architecture: :cnn,
        vector_length: 64,
        labels: ["negative", "positive"],
        decision_policy: %{strategy: :positive_threshold, positive_threshold: 0.0},
        minimum_input_tokens: 64
      )

    assert %{label: "positive", active_token_count: 64} = Nx.Serving.run(serving, text)
  end

  test "decision policy and minimum input validation are architecture-independent" do
    labels = %{0 => "negative", 1 => "positive"}
    argmax = %{strategy: :argmax}
    threshold = %{strategy: :positive_threshold, positive_threshold: 0.6}

    assert %{label: "negative"} =
             Serving.decode_prediction([0.5, 0.5], labels, argmax, 128, 0)

    assert %{label: "negative", probabilities: %{"positive" => 0.55}} =
             Serving.decode_prediction([0.45, 0.55], labels, threshold, 128, 0)

    assert %{label: "positive"} =
             Serving.decode_prediction([0.35, 0.65], labels, threshold, 128, 0)

    assert %{
             label: :insufficient_input,
             confidence: nil,
             probabilities: %{"negative" => 0.1, "positive" => 0.9}
           } = Serving.decode_prediction([0.1, 0.9], labels, threshold, 32, 64)
  end
end
