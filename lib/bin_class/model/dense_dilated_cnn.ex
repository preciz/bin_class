defmodule BinClass.Model.DenseDilatedCnn do
  @moduledoc false
  alias BinClass.Model.{Config, Masking}

  def build(vocab_size, opts) do
    opts = Config.validate_options!(vocab_size, opts, [])

    embedding_size = Keyword.fetch!(opts, :embedding_size)
    dropout_rate = Keyword.fetch!(opts, :dropout_rate)
    pad_token_id = Keyword.fetch!(opts, :pad_token_id)

    input = Axon.input("input")
    masking = Masking.new(input, pad_token_id)

    embedded =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> Masking.mask_sequence(masking)
      |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 0))

    f = 32

    # Block 1: Local n-gram multi-scale convolutions (1, 3, 5) -> 96 channels
    b1_1 = Axon.conv(embedded, f, kernel_size: 1, padding: :same, activation: :relu)
    b1_2 = Axon.conv(embedded, f, kernel_size: 3, padding: :same, activation: :relu)
    b1_3 = Axon.conv(embedded, f, kernel_size: 5, padding: :same, activation: :relu)

    block1 =
      Axon.concatenate([b1_1, b1_2, b1_3], axis: -1)
      |> Masking.mask_sequence(masking)

    # Block 2: Dilated multi-scale convolutions (dilations 1, 2, 4 with kernel 3) -> 96 channels
    b2_1 =
      Axon.conv(block1, f,
        kernel_size: 3,
        strides: [1],
        kernel_dilation: [1],
        padding: :same,
        activation: :relu
      )

    b2_2 =
      Axon.conv(block1, f,
        kernel_size: 3,
        strides: [1],
        kernel_dilation: [2],
        padding: :same,
        activation: :relu
      )

    b2_3 =
      Axon.conv(block1, f,
        kernel_size: 3,
        strides: [1],
        kernel_dilation: [4],
        padding: :same,
        activation: :relu
      )

    block2 =
      Axon.concatenate([b2_1, b2_2, b2_3], axis: -1)
      |> Masking.mask_sequence(masking)

    # Dense feature concatenation (192 channels) -> 4-head attentive pooling
    Axon.concatenate([block1, block2], axis: -1)
    |> Masking.masked_attentive_pooling(masking, heads: 4, hidden_size: 64)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 1))
    |> Axon.dense(2, activation: :softmax)
  end
end
