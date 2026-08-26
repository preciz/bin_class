defmodule BinClass.DocumentPredictorTest do
  use ExUnit.Case

  setup_all do
    training_text =
      Enum.join(
        ~w(alpha beta gamma delta epsilon zeta eta theta iota kappa lambda sentinel),
        " "
      )

    tokenizer = BinClass.Tokenizer.train([training_text])
    metadata = BinClass.Tokenizer.metadata(tokenizer)
    vector_length = 64

    model =
      BinClass.Model.build(:cnn, metadata.vocab_size, pad_token_id: metadata.pad_token_id)

    template = Nx.broadcast(metadata.pad_token_id, {1, vector_length}) |> Nx.as_type(:u16)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())

    classifier = %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: params,
      vector_length: vector_length,
      vocab_size: metadata.vocab_size,
      pad_token_id: metadata.pad_token_id,
      unk_token_id: metadata.unk_token_id,
      labels: %{0 => "negative", 1 => "positive"},
      architecture: :cnn,
      dropout_rate: 0.2,
      decision_policy: %{strategy: :argmax},
      minimum_input_tokens: 0
    }

    %{classifier: classifier}
  end

  test "returns predictions and exact original byte spans for every chunk", %{
    classifier: classifier
  } do
    text = "alpha beta gamma ☃ delta epsilon zeta eta theta iota kappa"

    predict =
      BinClass.compile_document_predictor(classifier,
        chunk_size: 8,
        chunk_overlap: 2,
        batch_size: 2
      )

    assert %{status: :ok, chunks: chunks} = predict.(text)
    assert length(chunks) > 1
    assert Enum.map(chunks, & &1.index) == Enum.to_list(0..(length(chunks) - 1))
    assert hd(chunks).start_byte == 0
    assert List.last(chunks).end_byte == byte_size(text)

    Enum.each(chunks, fn chunk ->
      assert chunk.text ==
               binary_part(text, chunk.start_byte, chunk.end_byte - chunk.start_byte)

      assert chunk.token_count > 0
      assert chunk.token_count <= 8
      assert chunk.active_token_count > 0
      assert chunk.active_token_count <= chunk.token_count
      assert chunk.prediction.active_token_count == chunk.active_token_count

      assert %{
               label: label,
               confidence: confidence,
               probabilities: %{"negative" => negative, "positive" => positive}
             } = chunk.prediction

      assert label in ["negative", "positive"]
      assert is_number(confidence)
      assert_in_delta negative + positive, 1.0, 1.0e-5
    end)

    chunks
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.each(fn [left, right] ->
      assert right.start_byte < left.end_byte
      assert right.end_byte > left.end_byte
    end)
  end

  test "scans complete long inputs without normalization or truncation", %{
    classifier: classifier
  } do
    text = String.duplicate("alpha ", 1_000) <> "sentinel"

    predict =
      BinClass.compile_document_predictor(classifier,
        chunk_size: 64,
        chunk_overlap: 0,
        batch_size: 64
      )

    assert %{status: :ok, chunks: chunks} = predict.(text)
    assert length(chunks) > 1
    assert List.last(chunks).end_byte == byte_size(text)
    assert String.ends_with?(List.last(chunks).text, "sentinel")
  end

  test "keeps insufficient chunks and raw probabilities", %{classifier: classifier} do
    classifier = %{classifier | minimum_input_tokens: 8}

    predict =
      BinClass.compile_document_predictor(classifier,
        chunk_size: 4,
        chunk_overlap: 1,
        batch_size: 2
      )

    assert %{status: :insufficient_input, chunks: [_ | _] = chunks} = predict.("alpha beta")

    assert Enum.all?(chunks, fn chunk ->
             chunk.prediction.label == :insufficient_input and
               chunk.prediction.confidence == nil and
               chunk.prediction.probabilities != %{}
           end)
  end

  test "literal padding tokens cannot satisfy document minimum input", %{
    classifier: classifier
  } do
    classifier = %{classifier | minimum_input_tokens: 1}
    predict = BinClass.compile_document_predictor(classifier, chunk_size: 8)

    assert %{status: :insufficient_input, chunks: [_ | _] = chunks} =
             predict.("[PAD] [PAD]")

    assert Enum.all?(chunks, &(&1.prediction.label == :insufficient_input))
  end

  test "returns no chunks for empty content and accepts only one document", %{
    classifier: classifier
  } do
    predict = BinClass.compile_document_predictor(classifier, chunk_size: 8)

    assert predict.("  \n\t") == %{status: :insufficient_input, chunks: []}

    assert_raise ArgumentError, ~r/expects a single string/, fn ->
      predict.(["first", "second"])
    end
  end

  test "validates chunk configuration before returning a predictor", %{classifier: classifier} do
    assert_raise ArgumentError, ~r/:chunk_size must be a positive integer/, fn ->
      BinClass.compile_document_predictor(classifier, chunk_size: 0)
    end

    assert_raise ArgumentError, ~r/no greater than.*vector length/, fn ->
      BinClass.compile_document_predictor(classifier, chunk_size: 65)
    end

    assert_raise ArgumentError, ~r/:chunk_overlap must be/, fn ->
      BinClass.compile_document_predictor(classifier, chunk_size: 8, chunk_overlap: 8)
    end
  end
end
