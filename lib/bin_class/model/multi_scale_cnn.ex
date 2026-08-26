defmodule BinClass.Model.MultiScaleCnn do
  @moduledoc false

  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 64)
    # Reduced filters per branch to maintain speed (Total 4 branches * 32 = 128 features)
    branch_filters = Keyword.get(opts, :branch_filters, 32)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)
    pad_token_id = Keyword.get(opts, :pad_token_id, 0)

    input = Axon.input("input")
    masking = BinClass.Model.Masking.new(input, pad_token_id)

    embedded =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> BinClass.Model.Masking.mask_sequence(masking)
      |> Axon.dropout(BinClass.Model.dropout_options(opts, dropout_rate, 0))

    # Branch A: 3-gram
    conv3 =
      embedded
      |> Axon.conv(branch_filters, kernel_size: 3, activation: :relu)

    # Branch B: 5-gram
    conv5 =
      embedded
      |> Axon.conv(branch_filters, kernel_size: 5, activation: :relu)

    # Mixed Pooling on both branches
    # This captures both the "strongest signal" (Max) and "average signal" (Avg)
    # for both short phrases (3-gram) and longer idioms (5-gram)
    features = [
      BinClass.Model.Masking.masked_global_max(conv3, masking),
      BinClass.Model.Masking.masked_global_avg(conv3, masking),
      BinClass.Model.Masking.masked_global_max(conv5, masking),
      BinClass.Model.Masking.masked_global_avg(conv5, masking)
    ]

    Axon.concatenate(features, axis: -1)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(BinClass.Model.dropout_options(opts, dropout_rate, 1))
    |> Axon.dense(2, activation: :softmax)
  end
end
