defmodule BinClass.Model.ConservativeCnn do
  @moduledoc false

  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 64)
    conv_filters = Keyword.get(opts, :conv_filters, 128)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)
    pad_token_id = Keyword.get(opts, :pad_token_id, 0)
    min_tokens = Keyword.get(opts, :min_tokens, 64)
    positive_logit_margin = Keyword.get(opts, :positive_logit_margin, 0.0)
    low_signal_penalty = Keyword.get(opts, :low_signal_penalty, 0.0)

    input = Axon.input("input")

    active_lengths =
      Axon.layer(
        fn x, _opts ->
          x
          |> Nx.not_equal(pad_token_id)
          |> Nx.sum(axes: [-1], keep_axes: true)
          |> Nx.as_type(:f32)
        end,
        [input]
      )

    logits =
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> Axon.dropout(rate: dropout_rate)
      |> Axon.conv(conv_filters, kernel_size: 3, activation: :relu)
      |> Axon.global_max_pool()
      |> Axon.dense(64, activation: :relu)
      |> Axon.dropout(rate: dropout_rate)
      |> Axon.dense(2)

    logits
    |> apply_conservative_bias(
      active_lengths,
      min_tokens,
      positive_logit_margin,
      low_signal_penalty
    )
    |> Axon.softmax()
  end

  defp apply_conservative_bias(
         logits,
         active_lengths,
         min_tokens,
         positive_logit_margin,
         low_signal_penalty
       ) do
    Axon.layer(
      fn logits, active_lengths, _opts ->
        batch_size = Nx.shape(logits) |> elem(0)

        low_signal? =
          active_lengths
          |> Nx.less(min_tokens)
          |> Nx.as_type(:f32)

        positive_penalty =
          low_signal?
          |> Nx.multiply(low_signal_penalty)
          |> Nx.add(positive_logit_margin)

        negative_logits = Nx.slice(logits, [0, 0], [batch_size, 1])

        positive_logits =
          logits
          |> Nx.slice([0, 1], [batch_size, 1])
          |> Nx.subtract(positive_penalty)

        Nx.concatenate([negative_logits, positive_logits], axis: -1)
      end,
      [logits, active_lengths]
    )
  end
end
