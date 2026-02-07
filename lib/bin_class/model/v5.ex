defmodule BinClass.Model.V5 do
  @moduledoc false

  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 64)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)

    input = Axon.input("input")

    embedded =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> Axon.dropout(rate: dropout_rate)

    # 3 Parallel branches with Kernels 1, 3, 5
    # 32 filters each (Total 96 filters)
    f = 32

    b1 = embedded |> Axon.conv(f, kernel_size: 1, padding: :same, activation: :relu)
    b2 = embedded |> Axon.conv(f, kernel_size: 3, padding: :same, activation: :relu)
    b3 = embedded |> Axon.conv(f, kernel_size: 5, padding: :same, activation: :relu)

    Axon.concatenate([b1, b2, b3], axis: -1)
    |> mixed_pooling()
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(rate: dropout_rate)
    |> Axon.dense(2, activation: :softmax)
  end

  defp mixed_pooling(x) do
    Axon.concatenate(
      [
        Axon.global_max_pool(x),
        Axon.global_avg_pool(x)
      ],
      axis: -1
    )
  end
end
