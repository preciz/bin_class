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
        vector_length: 16,
        embedding_size: 8,
        conv_filters: 4
      )

    assert %BinClass.Classifier{} = result
    assert result.tokenizer
    assert result.model_params
    assert result.accuracy
    assert result.epoch >= 0
    assert %{positive_threshold: _, min_positive_tokens: 64} = result.decision_policy

    BinClass.Tmp.with_tmp_dir(fn dir ->
      model_path = Path.join(dir, "model.bin")
      BinClass.save(result, model_path)

      loaded_classifier = BinClass.load_classifier(model_path)
      assert loaded_classifier.decision_policy == result.decision_policy

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

  test "deserializes classifiers saved before decision policy existed" do
    data = [
      %{text: "positive feeling good happy", label: 1},
      %{text: "negative feeling bad sad", label: 0}
    ]

    classifier = Trainer.train(data, epochs: 1, batch_size: 1, vector_length: 8)

    old_binary =
      classifier
      |> BinClass.serialize()
      |> :erlang.binary_to_term()
      |> Map.delete(:decision_policy)
      |> :erlang.term_to_binary()

    loaded = BinClass.deserialize_classifier(old_binary)
    assert %BinClass.Classifier{} = loaded
    assert loaded.decision_policy == nil
  end
end
