defmodule BinClass.TrainerAndServingTest do
  use ExUnit.Case
  alias BinClass.Trainer

  @tag timeout: 120_000
  test "trains a model and makes predictions" do
    # Create a dummy dataset
    data = [
      %{text: "positive feeling good happy", label: 1},
      %{text: "negative feeling bad sad", label: 0},
      %{text: "joy wonderful great", label: 1},
      %{text: "pain terrible awful", label: 0},
      %{text: "positive awesome", label: 1},
      %{text: "negative horrible", label: 0},
      %{text: "positive love", label: 1},
      %{text: "negative hate", label: 0},
      # Need enough data for split, let's add a few more to be safe for 90/10 split
      %{text: "good", label: 1},
      %{text: "bad", label: 0}
    ]

    result =
      Trainer.train(data,
        epochs: 2,
        batch_size: 2,
        vector_length: 16
      )

    assert %BinClass.Classifier{} = result
    assert result.tokenizer
    assert result.model_params
    assert result.accuracy
    assert result.epoch >= 0
    assert result.vocab_size == Tokenizers.Tokenizer.get_vocab_size(result.tokenizer)
    assert result.pad_token_id == Tokenizers.Tokenizer.token_to_id(result.tokenizer, "[PAD]")
    assert result.unk_token_id == Tokenizers.Tokenizer.token_to_id(result.tokenizer, "[UNK]")
    assert result.pad_token_id != result.unk_token_id
    assert result.decision_policy == %{strategy: :argmax}
    assert result.minimum_input_tokens == 0
    assert result.labels == ["0", "1"]

    BinClass.Tmp.with_tmp_dir(fn dir ->
      model_path = Path.join(dir, "model.bin")
      BinClass.save(result, model_path)

      loaded_classifier = BinClass.load_classifier(model_path)
      assert loaded_classifier.decision_policy == result.decision_policy
      assert loaded_classifier.minimum_input_tokens == result.minimum_input_tokens
      assert loaded_classifier.vocab_size == result.vocab_size
      assert loaded_classifier.pad_token_id == result.pad_token_id
      assert loaded_classifier.unk_token_id == result.unk_token_id

      serving = BinClass.load(model_path)

      prediction = Nx.Serving.run(serving, "happy")
      assert %{label: _, confidence: _} = prediction

      # Test loading with options
      serving_with_opts =
        BinClass.load(model_path, compiler: EXLA, defn_options: [compiler: EXLA])

      prediction2 = Nx.Serving.run(serving_with_opts, "happy")
      assert %{label: _, confidence: _} = prediction2

      batch_predictions = Nx.Serving.run(serving, ["happy", "sad"])
      assert length(batch_predictions) == 2
    end)
  end

  test "rejects classifiers that do not contain the complete serialized schema" do
    data = [
      %{text: "positive feeling good happy", label: 1},
      %{text: "negative feeling bad sad", label: 0}
    ]

    classifier = Trainer.train(data, epochs: 1, batch_size: 1, vector_length: 8)

    serialized_data =
      classifier
      |> BinClass.serialize()
      |> :erlang.binary_to_term()

    for field <- [
          :accuracy,
          :epoch,
          :learning_rate,
          :dropout_rate,
          :decision_policy,
          :minimum_input_tokens,
          :target_mode,
          :target_threshold
        ] do
      incomplete_binary =
        serialized_data
        |> Map.delete(field)
        |> :erlang.term_to_binary()

      assert_raise ArgumentError, "Serialized classifier is missing #{inspect(field)}", fn ->
        BinClass.deserialize_classifier(incomplete_binary)
      end
    end
  end

  @tag timeout: 120_000
  test "trains with soft targets and preserves target metadata through serialization" do
    data = [
      %{text: "completely negative experience", label: 0.0},
      %{text: "mostly unpleasant outcome", label: 0.1},
      %{text: "weakly negative feedback", label: 0.2},
      %{text: "mixed but leaning negative review", label: 0.3},
      %{text: "slightly disappointing result", label: 0.4},
      %{text: "somewhat positive experience", label: 0.6},
      %{text: "good result with minor concerns", label: 0.7},
      %{text: "very positive and satisfying outcome", label: 0.8},
      %{text: "excellent experience with strong praise", label: 0.9},
      %{text: "outstanding result exceeded every expectation", label: 1.0}
    ]

    classifier =
      Trainer.train(data,
        target_mode: :soft,
        target_threshold: 0.5,
        architecture: :cnn,
        epochs: 1,
        batch_size: 2,
        vector_length: 16
      )

    assert classifier.target_mode == :soft
    assert classifier.target_threshold == 0.5

    binary = BinClass.serialize(classifier)
    loaded = BinClass.deserialize_classifier(binary)

    assert loaded.target_mode == :soft
    assert loaded.target_threshold == 0.5
    assert loaded.architecture == :cnn

    serving = BinClass.deserialize(binary)
    assert %{label: label} = Nx.Serving.run(serving, "an excellent and satisfying result")
    assert label in ["0", "1"]
  end
end
