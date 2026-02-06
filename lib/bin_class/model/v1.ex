defmodule BinClass.Model.V1 do
  @moduledoc false

  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 64)
    conv_filters = Keyword.get(opts, :conv_filters, 128)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)

    Axon.input("input")
    |> Axon.embedding(vocab_size, embedding_size)
    |> Axon.dropout(rate: dropout_rate)
    |> Axon.conv(conv_filters, kernel_size: 3, activation: :relu)
    |> Axon.global_max_pool()
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(rate: dropout_rate)
    |> Axon.dense(2, activation: :softmax)
  end
end
