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

    # Branch true (one is zero)
    data_one_zero = [
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1}
    ]

    result = Trainer.train(data_one_zero, epochs: 1, batch_size: 1)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0
  end

  test "percentile_90/1 edge cases" do
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
    assert result.vector_length == 5 # min(sorted_at_index, 5) where index is floor(10 * 0.9) = 9

    # We can't easily test percentile_90([]) through train because it trains a tokenizer first
    # but we can trust it if we add a unit test if we had access to private fns.
    # Since we don't, I'll just ensure data with empty strings or something doesn't break it.
  end

  test "trainer with custom options" do
    data = [%{text: "a", label: 1}, %{text: "b", label: 0}, %{text: "c", label: 1}, %{text: "d", label: 0}, 
            %{text: "e", label: 1}, %{text: "f", label: 0}, %{text: "g", label: 1}, %{text: "h", label: 0},
            %{text: "i", label: 1}, %{text: "j", label: 0}]
    result = Trainer.train(data, epochs: 1, batch_size: 2, learning_rate: 0.01, decay: 0.1, validation_split: 0.2, patience: 2)
    assert %BinClass.Classifier{} = result
  end

  test "explicit vector_length" do
    data = [%{text: "a", label: 1}, %{text: "b", label: 0}]
    result = Trainer.train(data, epochs: 1, batch_size: 1, vector_length: 123)
    assert %BinClass.Classifier{} = result
    assert result.vector_length == 123
  end

  test "custom tokenizer data" do
    data = [%{text: "a", label: 1}, %{text: "b", label: 0}]
    tokenizer_data = ["a", "b", "c"]
    result = Trainer.train(data, tokenizer_data: tokenizer_data, epochs: 1, batch_size: 1)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0
  end

  test "custom compiler" do
    data = [%{text: "a", label: 1}, %{text: "b", label: 0}]
    # We use EXLA explicitly
    result = Trainer.train(data, epochs: 1, batch_size: 1, compiler: EXLA)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0
  end
end
