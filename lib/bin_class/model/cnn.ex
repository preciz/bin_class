defmodule BinClass.Model.Cnn do
  @moduledoc false

  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 64)
    conv_filters = Keyword.get(opts, :conv_filters, 128)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)
    pad_token_id = Keyword.get(opts, :pad_token_id, 0)

    input = Axon.input("input")
    masking = BinClass.Model.Masking.new(input, pad_token_id)

    features =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> BinClass.Model.Masking.mask_sequence(masking)
      |> Axon.dropout(BinClass.Model.dropout_options(opts, dropout_rate, 0))
      |> Axon.conv(conv_filters, kernel_size: 3, activation: :relu)

    features
    |> BinClass.Model.Masking.masked_global_max(
      BinClass.Model.Masking.valid_convolution(masking, 3)
    )
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(BinClass.Model.dropout_options(opts, dropout_rate, 1))
    |> Axon.dense(2, activation: :softmax)
  end
end
