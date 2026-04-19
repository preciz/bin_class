defmodule BinClass.ModelTest do
  use ExUnit.Case
  alias BinClass.Model

  test "builds cnn model" do
    vocab_size = 100
    model = Model.build(:cnn, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(:cnn, vocab_size, embedding_size: 32, conv_filters: 64, dropout_rate: 0.1)

    assert %Axon{} = model_opts

    # Direct call
    assert %Axon{} = BinClass.Model.Cnn.build(vocab_size)
  end

  test "builds cnn_mixed_pooling model" do
    vocab_size = 100
    model = Model.build(:cnn_mixed_pooling, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(:cnn_mixed_pooling, vocab_size,
        embedding_size: 32,
        conv_filters: 64,
        dropout_rate: 0.1
      )

    assert %Axon{} = model_opts

    # Direct call
    assert %Axon{} = BinClass.Model.CnnMixedPooling.build(vocab_size)
  end

  test "builds multi_scale_cnn model" do
    vocab_size = 100
    model = Model.build(:multi_scale_cnn, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(:multi_scale_cnn, vocab_size,
        embedding_size: 32,
        branch_filters: 16,
        dropout_rate: 0.1
      )

    assert %Axon{} = model_opts

    # Direct call
    assert %Axon{} = BinClass.Model.MultiScaleCnn.build(vocab_size)
  end

  test "builds sep_se_cnn model" do
    vocab_size = 100
    model = Model.build(:sep_se_cnn, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(:sep_se_cnn, vocab_size,
        embedding_size: 32,
        branch_filters: 16,
        dropout_rate: 0.1
      )

    assert %Axon{} = model_opts

    # Direct call
    assert %Axon{} = BinClass.Model.SepSeCnn.build(vocab_size)
  end

  test "builds parallel_cnn model" do
    vocab_size = 100
    model = Model.build(:parallel_cnn, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(:parallel_cnn, vocab_size, embedding_size: 32, dropout_rate: 0.1)

    assert %Axon{} = model_opts

    # Direct call
    assert %Axon{} = BinClass.Model.ParallelCnn.build(vocab_size)
  end

  test "BinClass.Model dispatcher default opts and backwards compatibility" do
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

    assert_raise ArgumentError, "Unknown model version: :unknown", fn ->
      Model.build(:unknown, 100)
    end
  end
end
