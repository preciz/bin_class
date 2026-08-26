defmodule BinClass.Model.Cnn do
  @moduledoc false
  alias BinClass.Model.Config

  def build(vocab_size, opts \\ []) do
    opts = Config.validate_options!(vocab_size, opts, conv_filters: 128)

    embedding_size = Keyword.fetch!(opts, :embedding_size)
    conv_filters = Keyword.fetch!(opts, :conv_filters)
    dropout_rate = Keyword.fetch!(opts, :dropout_rate)
    pad_token_id = Keyword.fetch!(opts, :pad_token_id)

    input = Axon.input("input")
    masking = BinClass.Model.Masking.new(input, pad_token_id)

    features =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> BinClass.Model.Masking.mask_sequence(masking)
      |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 0))
      |> Axon.conv(conv_filters, kernel_size: 3, activation: :relu)

    features
    |> BinClass.Model.Masking.masked_global_max(
      BinClass.Model.Masking.valid_convolution(masking, 3)
    )
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 1))
    |> Axon.dense(2, activation: :softmax)
  end
end
