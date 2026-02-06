defmodule BinClass.Model.V2 do
  @moduledoc false

  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 64)
    # Reduced from 128
    conv_filters = Keyword.get(opts, :conv_filters, 96)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)

    input = Axon.input("input")

    x =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> Axon.dropout(rate: dropout_rate)
      |> Axon.conv(conv_filters, kernel_size: 3, activation: :relu)

    # Global Average Pooling
    avg_pool = Axon.global_avg_pool(x)

    # Global Max Pooling
    max_pool = Axon.global_max_pool(x)

    Axon.concatenate([avg_pool, max_pool], axis: -1)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(rate: dropout_rate)
    |> Axon.dense(2, activation: :softmax)
  end
end
