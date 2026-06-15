defmodule BinClass.Model.Transformer do
  @moduledoc """
  A custom Transformer Encoder architecture with a Sequence Length Logit Bias
  to handle missing or extremely short text inputs.
  """

  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 64)
    ff_dim = Keyword.get(opts, :ff_dim, 128)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)
    pad_token_id = Keyword.get(opts, :pad_token_id, 0) # Pad token is 0 in Vectorizer output
    min_tokens = Keyword.get(opts, :min_tokens, 3)

    input = Axon.input("input")

    # 1. Calculate Active Token Length
    non_pad_mask = Axon.layer(fn x, _opts -> Nx.not_equal(x, pad_token_id) end, [input])
    
    active_lengths = Axon.layer(fn mask, _opts -> 
      Nx.sum(mask, axes: [-1], keep_axes: true) 
    end, [non_pad_mask])

    # 2. Main Transformer Layers
    embedded = 
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> Axon.dropout(rate: dropout_rate)

    # Self-attention Block
    # We project embedded to query, key, value using Dense layers
    q = Axon.dense(embedded, embedding_size)
    k = Axon.dense(embedded, embedding_size)
    v = Axon.dense(embedded, embedding_size)

    attn_output = Axon.layer(fn q_t, k_t, v_t, _opts ->
      # compute scores: [B, L, L] by contracting last axis (2) of q_t with last axis (2) of k_t
      scores = Nx.dot(q_t, [2], [0], k_t, [2], [0])
      scores_scaled = Nx.divide(scores, Nx.sqrt(embedding_size))
      attn_weights = softmax(scores_scaled, -1)
      # compute attention output: [B, L, E] by contracting axis 2 of attn_weights with axis 1 of v_t
      Nx.dot(attn_weights, [2], [0], v_t, [1], [0])
    end, [q, k, v])

    x = 
      Axon.add(embedded, attn_output)
      |> Axon.layer_norm()
      |> Axon.dropout(rate: dropout_rate)

    # Feed-forward network
    ffn_output = 
      x
      |> Axon.dense(ff_dim, activation: :relu)
      |> Axon.dense(embedding_size)
    
    x = 
      Axon.add(x, ffn_output)
      |> Axon.layer_norm()
      |> Axon.dropout(rate: dropout_rate)

    # Global Mixed Pooling (Average + Max)
    pooled = 
      Axon.concatenate(
        [
          Axon.global_max_pool(x),
          Axon.global_avg_pool(x)
        ],
        axis: -1
      )

    # Raw Logits layer
    logits = 
      pooled
      |> Axon.dense(64, activation: :relu)
      |> Axon.dropout(rate: dropout_rate)
      |> Axon.dense(2) # Output shape: [Batch, 2] (class 0, class 1)

    # 3. Apply Logit Bias
    logits_biased = Axon.layer(fn l, len, _opts ->
      is_short = Nx.less(len, min_tokens)
      penalty = Nx.multiply(is_short, 1000.0)
      
      # Split logits into class 0 and class 1
      logits_class_0 = Nx.slice(l, [0, 0], [Nx.shape(l) |> elem(0), 1])
      logits_class_1 = Nx.slice(l, [0, 1], [Nx.shape(l) |> elem(0), 1]) |> Nx.subtract(penalty)
      
      Nx.concatenate([logits_class_0, logits_class_1], axis: -1)
    end, [logits, active_lengths])

    # 4. Final Activation
    Axon.softmax(logits_biased)
  end

  defp softmax(logits, axis) do
    max_val = Nx.reduce_max(logits, axes: [axis], keep_axes: true)
    shifted_logits = Nx.subtract(logits, max_val)
    exps = Nx.exp(shifted_logits)
    sums = Nx.sum(exps, axes: [axis], keep_axes: true)
    Nx.divide(exps, sums)
  end
end
