defmodule BinClass.Model.MultiScaleCnn do
  @moduledoc false
  alias BinClass.Model.Config

  def build(vocab_size, opts \\ []) do
    opts = Config.validate_options!(vocab_size, opts, branch_filters: 32)

    embedding_size = Keyword.fetch!(opts, :embedding_size)
    branch_filters = Keyword.fetch!(opts, :branch_filters)
    dropout_rate = Keyword.fetch!(opts, :dropout_rate)
    pad_token_id = Keyword.fetch!(opts, :pad_token_id)

    input = Axon.input("input")
    masking = BinClass.Model.Masking.new(input, pad_token_id)

    embedded =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> BinClass.Model.Masking.mask_sequence(masking)
      |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 0))

    # Branch A: 3-gram
    conv3 =
      embedded
      |> Axon.conv(branch_filters, kernel_size: 3, activation: :relu)

    # Branch B: 5-gram
    conv5 =
      embedded
      |> Axon.conv(branch_filters, kernel_size: 5, activation: :relu)

    conv3_masking = BinClass.Model.Masking.valid_convolution(masking, 3)
    conv5_masking = BinClass.Model.Masking.valid_convolution(masking, 5)

    # Mixed Pooling on both branches
    # This captures both the "strongest signal" (Max) and "average signal" (Avg)
    # for both short phrases (3-gram) and longer idioms (5-gram)
    features = [
      BinClass.Model.Masking.masked_global_max(conv3, conv3_masking),
      BinClass.Model.Masking.masked_global_avg(conv3, conv3_masking),
      BinClass.Model.Masking.masked_global_max(conv5, conv5_masking),
      BinClass.Model.Masking.masked_global_avg(conv5, conv5_masking)
    ]

    Axon.concatenate(features, axis: -1)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 1))
    |> Axon.dense(2, activation: :softmax)
  end
end
