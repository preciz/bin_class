defmodule BinClass.TrainerTest do
  use ExUnit.Case
  alias BinClass.Trainer

  test "balance_data/1 covers all branches" do
    # Branch c1 > c0
    data_c1_gt_c0 = [
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "neg", label: 0}
    ]

    result = Trainer.train(data_c1_gt_c0, epochs: 1, batch_size: 1)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0

    # Branch c0 > c1
    data_c0_gt_c1 = [
      %{text: "neg", label: 0},
      %{text: "neg", label: 0},
      %{text: "pos", label: 1}
    ]

    result = Trainer.train(data_c0_gt_c1, epochs: 1, batch_size: 1)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0

    # Branch true (equal or one is zero)
    data_equal = [
      %{text: "neg", label: 0},
      %{text: "pos", label: 1}
    ]

    result = Trainer.train(data_equal, epochs: 1, batch_size: 1)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0
  end

  test "percentile_95/1 edge cases" do
    data = [
      %{text: "a", label: 1},
      %{text: "a", label: 1},
      %{text: "a", label: 1},
      %{text: "a", label: 1},
      %{text: "a", label: 1},
      %{text: "a", label: 1},
      %{text: "a", label: 1},
      %{text: "a", label: 1},
      %{text: "a", label: 1},
      %{text: "a", label: 1}
    ]

    result = Trainer.train(data, epochs: 1, batch_size: 1)
    assert %BinClass.Classifier{} = result
    assert result.vector_length > 0
  end

  test "custom tokenizer data" do
    data = [%{text: "a", label: 1}, %{text: "b", label: 0}]
    tokenizer_data = ["a", "b", "c"]
    result = Trainer.train(data, tokenizer_data: tokenizer_data, epochs: 1, batch_size: 1)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0
  end
end
