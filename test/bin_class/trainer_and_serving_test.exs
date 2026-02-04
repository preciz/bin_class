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

    BinClass.Tmp.with_tmp_dir(fn dir ->
      model_path = Path.join(dir, "model.bin")
      BinClass.save(result, model_path)

      serving = BinClass.load(model_path)

      prediction = Nx.Serving.run(serving, "happy")
      assert %{label: _, confidence: _} = prediction

      batch_predictions = Nx.Serving.run(serving, ["happy", "sad"])
      assert length(batch_predictions) == 2
    end)
  end
end
