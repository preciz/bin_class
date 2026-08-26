defmodule BinClass.ClassifierTest do
  use ExUnit.Case
  alias BinClass.Classifier

  test "inspect protocol" do
    classifier = %Classifier{
      architecture: :cnn,
      model_revision: BinClass.Model.revision(),
      labels: ["a", "b"],
      vector_length: 128,
      vocab_size: 1000,
      pad_token_id: 3,
      unk_token_id: 0,
      accuracy: 0.95,
      epoch: 5,
      learning_rate: 1.0e-3,
      dropout_rate: 0.2,
      decision_policy: %{strategy: :argmax},
      minimum_input_tokens: 12,
      target_mode: :hard,
      target_threshold: 0.5
    }

    inspected = inspect(classifier)
    assert inspected =~ "#BinClass.Classifier<"
    assert inspected =~ "architecture: :cnn"
    assert inspected =~ "accuracy: 0.95"
    assert inspected =~ ~s(labels: ["a", "b"])
    assert inspected =~ "vector_length: 128"
    assert inspected =~ "vocab_size: 1000"
    assert inspected =~ "pad_token_id: 3"
    assert inspected =~ "unk_token_id: 0"
    assert inspected =~ "epoch: 5"
    assert inspected =~ "decision_policy: %{strategy: :argmax}"
    assert inspected =~ "minimum_input_tokens: 12"

    # Ensure model_params (which could be huge) is NOT in the output
    refute inspected =~ "model_params"
  end
end
