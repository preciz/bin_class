defmodule BinClass.ModelTest do
  use ExUnit.Case
  alias BinClass.Model

  test "builds version 1 model" do
    vocab_size = 100
    model = Model.build(1, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(1, vocab_size, embedding_size: 32, conv_filters: 64, dropout_rate: 0.1)

    assert %Axon{} = model_opts

    # Direct call to V1
    assert %Axon{} = BinClass.Model.V1.build(vocab_size)
  end

  test "builds version 2 model" do
    vocab_size = 100
    model = Model.build(2, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(2, vocab_size, embedding_size: 32, conv_filters: 64, dropout_rate: 0.1)

    assert %Axon{} = model_opts

    # Direct call to V2
    assert %Axon{} = BinClass.Model.V2.build(vocab_size)
  end

  test "builds version 3 model" do
    vocab_size = 100
    model = Model.build(3, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(3, vocab_size, embedding_size: 32, branch_filters: 16, dropout_rate: 0.1)

    assert %Axon{} = model_opts

    # Direct call to V3
    assert %Axon{} = BinClass.Model.V3.build(vocab_size)
  end

  test "builds version 4 model" do
    vocab_size = 100
    model = Model.build(4, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(4, vocab_size, embedding_size: 32, branch_filters: 16, dropout_rate: 0.1)

    assert %Axon{} = model_opts

    # Direct call to V4
    assert %Axon{} = BinClass.Model.V4.build(vocab_size)
  end

  test "builds version 5 model" do
    vocab_size = 100
    model = Model.build(5, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(5, vocab_size, embedding_size: 32, dropout_rate: 0.1)

    assert %Axon{} = model_opts

    # Direct call to V5
    assert %Axon{} = BinClass.Model.V5.build(vocab_size)
  end

  test "BinClass.Model dispatcher default opts" do
    assert %Axon{} = Model.build(1, 100)
    assert %Axon{} = Model.build(2, 100)
    assert %Axon{} = Model.build(3, 100)
    assert %Axon{} = Model.build(4, 100)
    assert %Axon{} = Model.build(5, 100)
  end

  test "raises on unknown model version" do
    assert_raise ArgumentError, "Unknown model version: 99", fn ->
      Model.build(99, 100)
    end
  end
end
