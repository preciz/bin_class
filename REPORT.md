# High-Accuracy Transformer Model with Sequence Length Logit Bias

This document proposes and details the architecture for a new model option in the `BinClass` library. The proposed model prioritizes high classification accuracy over ultra-low inference latency, and includes a built-in mechanism to handle empty or extremely short inputs by leaning towards a classification of `0` (no match).

---

## 1. Model Selection: Transformer Encoder (Self-Attention)

To achieve maximum accuracy on text classification, we propose introducing a **Transformer Encoder** (Self-Attention) architecture.

### Why a Transformer Encoder?
* **Bidirectional Contextual Representation**: Unlike CNNs that look at fixed $N$-grams, the self-attention mechanism allows every token in the sequence to dynamically attend to every other token. This allows the model to learn complex dependencies and syntactic structures.
* **Flexibility with Sequence Structure**: It treats the inputs holistically, allowing it to capture subtle nuances, negations, and relationships across long text passages.
* **Inference vs. Accuracy Trade-off**: Self-attention is $O(N^2)$ relative to the sequence length. While this makes it computationally heavier and slightly slower than a 1D CNN, it is highly parallelizable and runs efficiently on modern accelerators (GPUs/TPUs) via the `EXLA` compiler.

---

## 2. Handling Missing or Short Text: Logit Bias Layer

A common failure mode of neural network text classifiers is predicting false positives (`1`) on empty or extremely short inputs due to dataset imbalance or activation bias. 

To ensure the model leans towards `0` (no match) when receiving too few tokens, we introduce a **Sequence Length Logit Bias** directly inside the Axon computation graph.

### Mathematical Formulation
Let:
* $x \in \mathbb{R}^{B \times L}$ be the input token IDs tensor for batch size $B$ and sequence length $L$.
* $pad\_id \in \mathbb{N}$ be the padding token ID (specifically `3` in the default tokenizer).
* $T_{min} \in \mathbb{N}$ be the minimum active token threshold (e.g., `3`).

We compute a boolean mask $M \in \{0, 1\}^{B \times L}$ indicating active (non-padded) tokens:
$$M_i = [x_i \neq pad\_id]$$

We sum the active tokens per sample to compute the sequence lengths $S \in \mathbb{R}^{B \times 1}$:
$$S_b = \sum_{j=1}^{L} M_{b,j}$$

We define the logit bias penalty $P \in \mathbb{R}^{B \times 1}$:
$$P_b = \begin{cases} 1000.0 & \text{if } S_b < T_{min} \\ 0.0 & \text{otherwise} \end{cases}$$

The final logit vector before the softmax layer is modified for class 1 (match):
$$z'_{b,0} = z_{b,0}$$
$$z'_{b,1} = z_{b,1} - P_b$$

When the active sequence length is below $T_{min}$, subtracting $1000.0$ from $z_{b,1}$ ensures that the softmax output for class 0 approaches $1.0$ ($100\%$ confidence).

---

## 3. Implementation in Axon

Below is the proposed Elixir implementation for `lib/bin_class/model/transformer.ex`:

```elixir
defmodule BinClass.Model.Transformer do
  @moduledoc """
  A custom Transformer Encoder architecture with a Sequence Length Logit Bias
  to handle missing or extremely short text inputs.
  """

  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 64)
    num_heads = Keyword.get(opts, :num_heads, 4)
    ff_dim = Keyword.get(opts, :ff_dim, 128)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)
    pad_token_id = Keyword.get(opts, :pad_token_id, 3) # [PAD] is index 3
    min_tokens = Keyword.get(opts, :min_tokens, 3)

    input = Axon.input("input")

    # 1. Calculate Active Token Length
    # mask is 1 where token != PAD, 0 where token == PAD
    non_pad_mask = Axon.layer(fn x, _opts -> Nx.not_equal(x, pad_token_id) end, [input])
    
    # active_lengths: [Batch, 1] representing number of non-pad tokens
    active_lengths = Axon.layer(fn mask, _opts -> 
      Nx.sum(mask, axes: [-1], keep_axes: true) 
    end, [non_pad_mask])

    # 2. Main Transformer Layers
    embedded = 
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> Axon.dropout(rate: dropout_rate)

    # Self-attention Block
    attn_output = Axon.multi_head_attention(embedded, num_heads, [query: embedded, key: embedded, value: embedded])
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
    # If active_lengths < min_tokens, subtract a huge penalty from the class 1 logit
    logits_biased = Axon.layer(fn [l, len], _opts ->
      # is_short is 1.0 where active length < threshold, 0.0 otherwise
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
end
```
