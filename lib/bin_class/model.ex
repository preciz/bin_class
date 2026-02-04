defmodule BinClass.Model do
  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 32)
    conv_filters = Keyword.get(opts, :conv_filters, 32)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.5)

    input = Axon.input("input")

    embedding =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> Axon.spatial_dropout(rate: 0.1)
      |> Axon.layer_norm()

    branches =
      for kernel_size <- [3, 4, 5] do
        embedding
        |> Axon.conv(conv_filters, kernel_size: kernel_size, activation: :relu, padding: :same)
        |> Axon.global_max_pool()
      end

    Axon.concatenate(branches)
    |> Axon.flatten()
    |> Axon.layer_norm()
    |> Axon.dropout(rate: dropout_rate)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(rate: dropout_rate)
    |> Axon.dense(2, activation: :softmax)
  end
end
