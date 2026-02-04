defmodule BinClass.ClassifierTest do
  use ExUnit.Case
  alias BinClass.Classifier

  test "inspect protocol" do
    classifier = %Classifier{
      labels: [:a, :b],
      vector_length: 128,
      vocab_size: 1000,
      accuracy: 0.95,
      epoch: 5
    }

    inspected = inspect(classifier)
    assert inspected =~ "#BinClass.Classifier<"
    assert inspected =~ "accuracy: 0.95"
    assert inspected =~ "labels: [:a, :b]"
    assert inspected =~ "vector_length: 128"
    assert inspected =~ "vocab_size: 1000"
    assert inspected =~ "epoch: 5"
    
    # Ensure model_params (which could be huge) is NOT in the output
    refute inspected =~ "model_params"
  end
end
