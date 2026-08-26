defmodule BinClass.Model.CnnMixedPooling do
  @moduledoc false
  alias BinClass.Model.Config

  def build(vocab_size, opts \\ []) do
    opts = Config.validate_options!(vocab_size, opts, conv_filters: 96)

    embedding_size = Keyword.fetch!(opts, :embedding_size)
    conv_filters = Keyword.fetch!(opts, :conv_filters)
    dropout_rate = Keyword.fetch!(opts, :dropout_rate)
    pad_token_id = Keyword.fetch!(opts, :pad_token_id)

    input = Axon.input("input")
    masking = BinClass.Model.Masking.new(input, pad_token_id)

    x =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> BinClass.Model.Masking.mask_sequence(masking)
      |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 0))
      |> Axon.conv(conv_filters, kernel_size: 3, activation: :relu)

    convolution_masking = BinClass.Model.Masking.valid_convolution(masking, 3)

    # Global Average Pooling
    avg_pool = BinClass.Model.Masking.masked_global_avg(x, convolution_masking)

    # Global Max Pooling
    max_pool = BinClass.Model.Masking.masked_global_max(x, convolution_masking)

    Axon.concatenate([avg_pool, max_pool], axis: -1)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 1))
    |> Axon.dense(2, activation: :softmax)
  end
end
