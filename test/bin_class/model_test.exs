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

  test "builds transformer model" do
    vocab_size = 100
    model = Model.build(:transformer, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(:transformer, vocab_size, embedding_size: 32, dropout_rate: 0.1)

    assert %Axon{} = model_opts

    # Direct call
    assert %Axon{} = BinClass.Model.Transformer.build(vocab_size)
  end

  test "transformer model execution and logit bias penalty" do
    vocab_size = 50
    model = BinClass.Model.Transformer.build(vocab_size,
      embedding_size: 8,
      ff_dim: 16,
      min_tokens: 3,
      dropout_rate: 0.0
    )

    {init_fn, predict_fn} = Axon.build(model)
    template = Nx.broadcast(0, {1, 5}) |> Nx.as_type(:u16)
    params = init_fn.(template, Axon.ModelState.empty())

    # Case 1: Short input (only 1 active token [1, 0, 0, 0, 0])
    short_input = Nx.tensor([[1, 0, 0, 0, 0]], type: :u16)
    preds_short = predict_fn.(params, short_input)
    assert Nx.shape(preds_short) == {1, 2}

    # For short inputs, class 1 is penalized by 1000.0, so class 0 probability should be ~1.0
    [p0, p1] = preds_short[0] |> Nx.to_list()
    assert p0 > 0.999
    assert p1 < 0.001

    # Case 2: Long input (4 active tokens [1, 2, 3, 4, 0])
    long_input = Nx.tensor([[1, 2, 3, 4, 0]], type: :u16)
    preds_long = predict_fn.(params, long_input)
    assert Nx.shape(preds_long) == {1, 2}

    # No penalty, so probabilities should be valid and positive
    [lp0, lp1] = preds_long[0] |> Nx.to_list()
    assert lp0 > 0.0
    assert lp1 > 0.0
  end

  test "BinClass.Model dispatcher default opts and backwards compatibility" do
    assert %Axon{} = Model.build(1, 100)
    assert %Axon{} = Model.build(2, 100)
    assert %Axon{} = Model.build(3, 100)
    assert %Axon{} = Model.build(4, 100)
    assert %Axon{} = Model.build(5, 100)
    assert %Axon{} = Model.build(6, 100)
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
