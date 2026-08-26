defmodule BinClass.Model.ParallelCnn do
  @moduledoc false
  alias BinClass.Model.Config

  def build(vocab_size, opts \\ []) do
    opts = Config.validate_options!(vocab_size, opts)

    embedding_size = Keyword.fetch!(opts, :embedding_size)
    dropout_rate = Keyword.fetch!(opts, :dropout_rate)
    pad_token_id = Keyword.fetch!(opts, :pad_token_id)

    input = Axon.input("input")
    masking = BinClass.Model.Masking.new(input, pad_token_id)

    embedded =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> BinClass.Model.Masking.mask_sequence(masking)
      |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 0))

    # 3 Parallel branches with Kernels 1, 3, 5
    # 32 filters each (Total 96 filters)
    f = 32

    b1 = embedded |> Axon.conv(f, kernel_size: 1, padding: :same, activation: :relu)
    b2 = embedded |> Axon.conv(f, kernel_size: 3, padding: :same, activation: :relu)
    b3 = embedded |> Axon.conv(f, kernel_size: 5, padding: :same, activation: :relu)

    Axon.concatenate([b1, b2, b3], axis: -1)
    |> mixed_pooling(masking)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 1))
    |> Axon.dense(2, activation: :softmax)
  end

  defp mixed_pooling(x, masking) do
    Axon.concatenate(
      [
        BinClass.Model.Masking.masked_global_max(x, masking),
        BinClass.Model.Masking.masked_global_avg(x, masking)
      ],
      axis: -1
    )
  end
end
