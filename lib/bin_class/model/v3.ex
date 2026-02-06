defmodule BinClass.Model.V3 do
  @moduledoc false

  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 64)
    # Reduced filters per branch to maintain speed (Total 4 branches * 32 = 128 features)
    branch_filters = Keyword.get(opts, :branch_filters, 32) 
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)

    input = Axon.input("input")
    
    embedded = 
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> Axon.dropout(rate: dropout_rate)

    # Branch A: 3-gram
    conv3 = 
      embedded
      |> Axon.conv(branch_filters, kernel_size: 3, activation: :relu)

    # Branch B: 5-gram
    conv5 = 
      embedded
      |> Axon.conv(branch_filters, kernel_size: 5, activation: :relu)

    # Mixed Pooling on both branches
    # This captures both the "strongest signal" (Max) and "average signal" (Avg)
    # for both short phrases (3-gram) and longer idioms (5-gram)
    features = [
      Axon.global_max_pool(conv3),
      Axon.global_avg_pool(conv3),
      Axon.global_max_pool(conv5),
      Axon.global_avg_pool(conv5)
    ]

    Axon.concatenate(features, axis: -1)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(rate: dropout_rate)
    |> Axon.dense(2, activation: :softmax)
  end
end
