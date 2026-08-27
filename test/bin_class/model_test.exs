defmodule BinClass.ModelTest do
  use ExUnit.Case
  alias BinClass.Model

  alias BinClass.Model.{
    Cnn,
    CnnMixedPooling,
    Config,
    Masking,
    MultiScaleCnn,
    ParallelCnn,
    SepSeCnn,
    Transformer
  }

  test "builds cnn model" do
    vocab_size = 100
    model = Model.build(:cnn, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(:cnn, vocab_size, embedding_size: 32, conv_filters: 64, dropout_rate: 0.1)

    assert %Axon{} = model_opts
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
  end

  test "builds parallel_cnn model" do
    vocab_size = 100
    model = Model.build(:parallel_cnn, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(:parallel_cnn, vocab_size, embedding_size: 32, dropout_rate: 0.1)

    assert %Axon{} = model_opts
  end

  test "builds transformer model" do
    vocab_size = 100
    model = Model.build(:transformer, vocab_size)
    assert %Axon{} = model

    # Test custom opts
    model_opts =
      Model.build(:transformer, vocab_size, embedding_size: 32, dropout_rate: 0.1)

    assert %Axon{} = model_opts
  end

  test "transformer model execution does not impose an input-length decision" do
    vocab_size = 50

    model =
      Transformer.build(vocab_size,
        embedding_size: 8,
        ff_dim: 16,
        dropout_rate: 0.0
      )

    {init_fn, predict_fn} = Axon.build(model)
    template = Nx.broadcast(0, {1, 5}) |> Nx.as_type(:u16)
    params = init_fn.(template, Axon.ModelState.empty())

    short_input = Nx.tensor([[1, 0, 0, 0, 0]], type: :u16)
    preds_short = predict_fn.(params, short_input)
    assert Nx.shape(preds_short) == {1, 2}

    [p0, p1] = preds_short[0] |> Nx.to_list()
    assert_in_delta p0 + p1, 1.0, 1.0e-6
    assert p0 > 0.0
    assert p1 > 0.0

    long_input = Nx.tensor([[1, 2, 3, 4, 0]], type: :u16)
    preds_long = predict_fn.(params, long_input)
    assert Nx.shape(preds_long) == {1, 2}

    [lp0, lp1] = preds_long[0] |> Nx.to_list()
    assert_in_delta lp0 + lp1, 1.0, 1.0e-6
    assert lp0 > 0.0
    assert lp1 > 0.0
  end

  test "lists all named model architectures" do
    assert Model.architectures() == [
             :cnn,
             :cnn_mixed_pooling,
             :multi_scale_cnn,
             :sep_se_cnn,
             :parallel_cnn,
             :transformer
           ]
  end

  test "reports and validates architecture-specific minimum vector lengths" do
    assert Model.minimum_vector_length(:cnn) == 3
    assert Model.minimum_vector_length(:cnn_mixed_pooling) == 3
    assert Model.minimum_vector_length(:multi_scale_cnn) == 5

    for architecture <- [:sep_se_cnn, :parallel_cnn, :transformer] do
      assert Model.minimum_vector_length(architecture) == 1
      assert Model.validate_vector_length!(architecture, 1) == 1
    end

    assert_raise ArgumentError, ~r/must be at least 3 for architecture :cnn/, fn ->
      Model.validate_vector_length!(:cnn, 2)
    end

    assert_raise ArgumentError, ~r/must be at least 5 for architecture :multi_scale_cnn/, fn ->
      Model.validate_vector_length!(:multi_scale_cnn, 4)
    end
  end

  test "all architectures ignore right-padding, including after exact-width input" do
    pad_token_id = 3
    active_ids = [11, 0, 13, 14, 15, 16, 17, 18]

    input = fn vector_length ->
      [active_ids ++ List.duplicate(pad_token_id, vector_length - length(active_ids))]
      |> Nx.tensor(type: :u16)
    end

    for architecture <- Model.architectures() do
      model =
        Model.build(
          architecture,
          100,
          architecture_options(architecture,
            dropout_rate: 0.0,
            pad_token_id: pad_token_id,
            embedding_size: 16,
            seed: 1234
          )
        )

      {init_fn, predict_fn} = Axon.build(model, seed: 42)
      params = init_fn.(input.(128), Axon.ModelState.empty())

      exact_prediction = predict_fn.(params, input.(length(active_ids)))
      short_prediction = predict_fn.(params, input.(16))
      long_prediction = predict_fn.(params, input.(128))

      assert Nx.all_close(exact_prediction, short_prediction, atol: 1.0e-6, rtol: 1.0e-6)
             |> Nx.to_number() == 1,
             "#{inspect(architecture)} changed its prediction when padding was first added"

      assert Nx.all_close(short_prediction, long_prediction, atol: 1.0e-6, rtol: 1.0e-6)
             |> Nx.to_number() == 1,
             "#{inspect(architecture)} changed its prediction when only padding was added"

      fully_padded = Nx.broadcast(pad_token_id, {1, 16}) |> Nx.as_type(:u16)
      fully_padded_prediction = predict_fn.(params, fully_padded)

      assert Nx.equal(fully_padded_prediction, fully_padded_prediction)
             |> Nx.all()
             |> Nx.to_number() == 1,
             "#{inspect(architecture)} produced NaN for a fully padded input"

      assert Nx.all_close(Nx.sum(fully_padded_prediction), Nx.tensor(1.0))
             |> Nx.to_number() == 1
    end
  end

  test "all architectures preserve exact padding holes when more right-padding is added" do
    pad_token_id = 3
    token_ids = [11, 12, 13, pad_token_id, 15, 16, 17, 18]

    input = fn vector_length ->
      [token_ids ++ List.duplicate(pad_token_id, vector_length - length(token_ids))]
      |> Nx.tensor(type: :u16)
    end

    for architecture <- Model.architectures() do
      model =
        Model.build(
          architecture,
          100,
          architecture_options(architecture,
            dropout_rate: 0.0,
            pad_token_id: pad_token_id,
            embedding_size: 16,
            seed: 1234
          )
        )

      {init_fn, predict_fn} = Axon.build(model, seed: 42)
      params = init_fn.(input.(128), Axon.ModelState.empty())

      exact_prediction = predict_fn.(params, input.(length(token_ids)))
      padded_prediction = predict_fn.(params, input.(128))

      assert Nx.all_close(exact_prediction, padded_prediction, atol: 1.0e-6, rtol: 1.0e-6)
             |> Nx.to_number() == 1,
             "#{inspect(architecture)} reconstructed a padding hole as an active position"
    end
  end

  test "raises on removed, numeric, and unknown model architectures" do
    assert_raise ArgumentError, "Unknown model architecture: :conservative_cnn", fn ->
      Model.build(:conservative_cnn, 100)
    end

    assert_raise ArgumentError, "Unknown model architecture: 1", fn ->
      Model.build(1, 100)
    end

    assert_raise ArgumentError, "Unknown model architecture: 99", fn ->
      Model.build(99, 100)
    end

    assert_raise ArgumentError, "Unknown model architecture: :unknown", fn ->
      Model.build(:unknown, 100)
    end
  end

  test "dispatcher, direct model builders, and masking reject unknown options" do
    modules = %{
      cnn: Cnn,
      cnn_mixed_pooling: CnnMixedPooling,
      multi_scale_cnn: MultiScaleCnn,
      sep_se_cnn: SepSeCnn,
      parallel_cnn: ParallelCnn,
      transformer: Transformer
    }

    for {architecture, module} <- modules do
      assert_raise ArgumentError, ~r/unknown keys.*:not_an_option/s, fn ->
        Model.build(architecture, 100, not_an_option: true)
      end

      assert_raise ArgumentError, ~r/unknown keys.*:not_an_option/s, fn ->
        module.build(100, not_an_option: true)
      end
    end

    input = Axon.input("input")
    masking = Masking.new(input, 0)

    assert_raise ArgumentError, ~r/unknown keys.*:not_an_option/s, fn ->
      Masking.masked_global_avg(input, masking, not_an_option: true)
    end
  end

  test "model builders and masking reject invalid option values" do
    for architecture <- Model.architectures() do
      for {opts, message} <- [
            {[embedding_size: 0], ~r/:embedding_size must be a positive integer/},
            {[dropout_rate: -0.1], ~r/:dropout_rate must be a number/},
            {[dropout_rate: 1], ~r/:dropout_rate must be a number/},
            {[pad_token_id: -1], ~r/:pad_token_id must be a non-negative integer/},
            {[pad_token_id: 100], ~r/:pad_token_id must be below :vocab_size/},
            {[seed: :random], ~r/:seed must be an integer or nil/}
          ] do
        assert_raise ArgumentError, message, fn -> Model.build(architecture, 100, opts) end
      end

      assert_raise ArgumentError, ~r/:vocab_size must be a positive integer/, fn ->
        Model.build(architecture, 0)
      end
    end

    for {architecture, option} <- [
          {:cnn, :conv_filters},
          {:cnn_mixed_pooling, :conv_filters},
          {:multi_scale_cnn, :branch_filters},
          {:sep_se_cnn, :branch_filters},
          {:transformer, :ff_dim}
        ] do
      assert_raise ArgumentError, ~r/#{option} must be a positive integer/, fn ->
        Model.build(architecture, 100, [{option, 0}])
      end
    end

    assert_raise ArgumentError, ~r/:embedding_size must be a positive integer/, fn ->
      Cnn.build(100, embedding_size: 0)
    end

    assert_raise ArgumentError, ~r/:seed must be an integer or nil/, fn ->
      Config.dropout_options(:random, 0.2, 0)
    end

    input = Axon.input("input")
    masking = Masking.new(input, 0)

    assert_raise ArgumentError, ~r/:keep_axes must be a boolean/, fn ->
      Masking.masked_global_avg(input, masking, keep_axes: :yes)
    end
  end

  test "kernel-one convolution masking preserves the exact token mask" do
    input = Axon.input("input")

    %Masking{token_mask: output} =
      input
      |> Masking.new(3)
      |> Masking.valid_convolution(1)

    {init_fn, predict_fn} = Axon.build(output)
    token_ids = Nx.tensor([[1, 3, 0, 3]], type: :u16)
    params = init_fn.(token_ids, Axon.ModelState.empty())

    assert predict_fn.(params, token_ids) |> Nx.to_list() == [[1, 0, 1, 0]]
  end

  test "masked_attentive_pooling applies attention weights across active tokens" do
    input = Axon.input("input")
    masking = Masking.new(input, 0)
    embedded = Axon.embedding(input, 50, 16)

    pooled = Masking.masked_attentive_pooling(embedded, masking, heads: 2, hidden_size: 16)
    {init_fn, predict_fn} = Axon.build(pooled)

    token_ids = Nx.tensor([[1, 2, 0, 0], [1, 2, 3, 4]], type: :u16)
    params = init_fn.(token_ids, Axon.ModelState.empty())
    result = predict_fn.(params, token_ids)

    # 2 heads * 16 channels = 32 output channels
    assert Nx.shape(result) == {2, 32}
    refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
  end

  defp architecture_options(architecture, common) do
    architecture_specific =
      case architecture do
        architecture when architecture in [:cnn, :cnn_mixed_pooling] -> [conv_filters: 16]
        architecture when architecture in [:multi_scale_cnn, :sep_se_cnn] -> [branch_filters: 8]
        :parallel_cnn -> []
        :transformer -> [ff_dim: 32]
      end

    Keyword.merge(common, architecture_specific)
  end
end
