defmodule BinClass.Model.CnnMixedPooling do
  @moduledoc false

  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 64)
    # Reduced from 128
    conv_filters = Keyword.get(opts, :conv_filters, 96)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)
    pad_token_id = Keyword.get(opts, :pad_token_id, 0)

    input = Axon.input("input")
    masking = BinClass.Model.Masking.new(input, pad_token_id)

    x =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> BinClass.Model.Masking.mask_sequence(masking)
      |> Axon.dropout(BinClass.Model.dropout_options(opts, dropout_rate, 0))
      |> Axon.conv(conv_filters, kernel_size: 3, activation: :relu)

    # Global Average Pooling
    avg_pool = BinClass.Model.Masking.masked_global_avg(x, masking)

    # Global Max Pooling
    max_pool = BinClass.Model.Masking.masked_global_max(x, masking)

    Axon.concatenate([avg_pool, max_pool], axis: -1)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(BinClass.Model.dropout_options(opts, dropout_rate, 1))
    |> Axon.dense(2, activation: :softmax)
  end
end
