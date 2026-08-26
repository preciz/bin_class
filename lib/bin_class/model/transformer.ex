defmodule BinClass.Model.Transformer do
  @moduledoc false
  alias BinClass.Model.{Config, Masking}

  def build(vocab_size, opts) do
    opts = Config.validate_options!(vocab_size, opts, ff_dim: 128)

    embedding_size = Keyword.fetch!(opts, :embedding_size)
    ff_dim = Keyword.fetch!(opts, :ff_dim)
    dropout_rate = Keyword.fetch!(opts, :dropout_rate)
    pad_token_id = Keyword.fetch!(opts, :pad_token_id)

    input = Axon.input("input")
    masking = Masking.new(input, pad_token_id)

    embedded_raw = Axon.embedding(input, vocab_size, embedding_size)

    embedded =
      Axon.layer(
        fn emb, _opts ->
          {_batch, seq_len, emb_size} = Nx.shape(emb)
          pos = Nx.iota({seq_len, 1}) |> Nx.as_type(:f32)

          div_t =
            Nx.iota({1, emb_size})
            |> Nx.as_type(:f32)
            |> Nx.divide(2)
            |> Nx.multiply(2)
            |> Nx.divide(emb_size)
            |> then(fn p -> Nx.pow(10_000.0, p) end)

          angle = Nx.divide(pos, div_t)

          even =
            Nx.iota({1, emb_size})
            |> Nx.remainder(2)
            |> Nx.equal(0)
            |> Nx.broadcast({seq_len, emb_size})

          pos_emb = Nx.select(even, Nx.sin(angle), Nx.cos(angle)) |> Nx.new_axis(0)
          Nx.add(emb, pos_emb)
        end,
        [embedded_raw]
      )
      |> Masking.mask_sequence(masking)
      |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 0))

    # Self-attention Block
    # We project embedded to query, key, value using Dense layers
    q = Axon.dense(embedded, embedding_size)
    k = Axon.dense(embedded, embedding_size)
    v = Axon.dense(embedded, embedding_size)

    attention_scores =
      Axon.layer(
        fn q_t, k_t, _opts ->
          # compute scores: [B, L, L] by contracting last axis (2) of q_t with last axis (2) of k_t
          scores = Nx.dot(q_t, [2], [0], k_t, [2], [0])
          scores_scaled = Nx.divide(scores, Nx.sqrt(embedding_size))
          scores_scaled
        end,
        [q, k]
      )

    masked_scores =
      Masking.mask_attention_scores(attention_scores, masking)

    attn_output =
      Axon.layer(
        fn scores_scaled, v_t, _opts ->
          attn_weights = softmax(scores_scaled, -1)

          # compute attention output: [B, L, E] by contracting axis 2 of attn_weights with axis 1 of v_t
          Nx.dot(attn_weights, [2], [0], v_t, [1], [0])
        end,
        [masked_scores, v]
      )

    x =
      Axon.add(embedded, attn_output)
      |> Axon.layer_norm()
      |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 1))

    # Feed-forward network
    ffn_output =
      x
      |> Axon.dense(ff_dim, activation: :relu)
      |> Axon.dense(embedding_size)

    x =
      Axon.add(x, ffn_output)
      |> Axon.layer_norm()
      |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 2))

    # Global Mixed Pooling (Average + Max)
    pooled =
      Axon.concatenate(
        [
          Masking.masked_global_max(x, masking),
          Masking.masked_global_avg(x, masking)
        ],
        axis: -1
      )

    pooled
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(Config.dropout_options(Keyword.get(opts, :seed), dropout_rate, 3))
    |> Axon.dense(2)
    |> Axon.softmax()
  end

  defp softmax(logits, axis) do
    max_val = Nx.reduce_max(logits, axes: [axis], keep_axes: true)
    shifted_logits = Nx.subtract(logits, max_val)
    exps = Nx.exp(shifted_logits)
    sums = Nx.sum(exps, axes: [axis], keep_axes: true)
    Nx.divide(exps, sums)
  end
end
