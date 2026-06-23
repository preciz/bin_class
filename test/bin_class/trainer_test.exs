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

    result = Trainer.train(data_c1_gt_c0, epochs: 1, batch_size: 1, vector_length: 8)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0

    # Branch c0 > c1
    data_c0_gt_c1 = [
      %{text: "neg", label: 0},
      %{text: "neg", label: 0},
      %{text: "pos", label: 1}
    ]

    result = Trainer.train(data_c0_gt_c1, epochs: 1, batch_size: 1, vector_length: 8)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0

    # Branch true (equal or one is zero)
    data_equal = [
      %{text: "neg", label: 0},
      %{text: "pos", label: 1}
    ]

    result = Trainer.train(data_equal, epochs: 1, batch_size: 1, vector_length: 8)
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

    result = Trainer.train(data_one_zero, epochs: 1, batch_size: 1, vector_length: 8)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0
  end

  test "default vector_length is 512" do
    data = [
      %{text: "a", label: 1},
      %{text: "b", label: 0}
    ]

    result = Trainer.train(data, epochs: 1, batch_size: 1)
    assert %BinClass.Classifier{} = result
    assert result.vector_length == 512
  end

  test "trainer with custom options" do
    data = [
      %{text: "a", label: 1},
      %{text: "b", label: 0},
      %{text: "c", label: 1},
      %{text: "d", label: 0},
      %{text: "e", label: 1},
      %{text: "f", label: 0},
      %{text: "g", label: 1},
      %{text: "h", label: 0},
      %{text: "i", label: 1},
      %{text: "j", label: 0}
    ]

    result =
      Trainer.train(data,
        epochs: 1,
        batch_size: 2,
        learning_rate: 0.01,
        decay: 0.1,
        validation_split: 0.2,
        patience: 2,
        vector_length: 8
      )

    assert %BinClass.Classifier{} = result
  end

  test "hyperparameter auto-tuning" do
    data = [
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "neg", label: 0},
      %{text: "neg", label: 0},
      %{text: "neg", label: 0},
      %{text: "neg", label: 0},
      %{text: "neg", label: 0},
      %{text: "neg", label: 0},
      %{text: "neg", label: 0},
      %{text: "neg", label: 0},
      %{text: "neg", label: 0},
      %{text: "neg", label: 0}
    ]

    # We use a very small subset for tuning in the test to keep it fast
    result = Trainer.train(data, epochs: 1, batch_size: 2, tune: true, vector_length: 8)
    assert %BinClass.Classifier{} = result
    assert result.learning_rate in [1.0e-2, 1.0e-3, 5.0e-4, 1.0e-4]
    assert result.dropout_rate in [0.1, 0.2, 0.3, 0.4, 0.5]
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

    result =
      Trainer.train(data,
        tokenizer_data: tokenizer_data,
        epochs: 1,
        batch_size: 1,
        vector_length: 8
      )

    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0
  end

  test "custom compiler" do
    data = [%{text: "a", label: 1}, %{text: "b", label: 0}]
    # We use EXLA explicitly
    result = Trainer.train(data, epochs: 1, batch_size: 1, compiler: EXLA, vector_length: 8)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0
  end
end
