defmodule BinClass.Model.V4 do
  @moduledoc false

  def build(vocab_size, opts \\ []) do
    embedding_size = Keyword.get(opts, :embedding_size, 64)
    # Filters for the pointwise convolution (projection)
    # Increased to 48 because SepConv is cheap, so we can afford more channels than V3 (32)
    branch_filters = Keyword.get(opts, :branch_filters, 48) 
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.2)

    input = Axon.input("input")
    
    embedded = 
      input
      |> Axon.embedding(vocab_size, embedding_size)
      |> Axon.dropout(rate: dropout_rate)

    # Branches for 3, 4, 5-grams
    branches = 
      [3, 4, 5]
      |> Enum.map(fn k -> 
        embedded
        |> depthwise_separable_conv(k, embedding_size, branch_filters)
        |> se_block(branch_filters)
        |> mixed_pooling()
      end)

    Axon.concatenate(branches, axis: -1)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(rate: dropout_rate)
    |> Axon.dense(2, activation: :softmax)
  end

  defp depthwise_separable_conv(x, kernel_size, input_channels, output_channels) do
    # 1. Depthwise Convolution
    # feature_group_size: 1 means each filter looks at only 1 input channel.
    # We set units (output channels) = input_channels to maintain 1-to-1 mapping.
    depthwise = 
      x
      |> Axon.conv(input_channels, 
          kernel_size: kernel_size, 
          feature_group_size: 1, 
          padding: :same, 
          use_bias: false
         )

    # 2. Pointwise Convolution
    # 1x1 conv to mix channels and project to output_channels
    depthwise
    |> Axon.conv(output_channels, kernel_size: 1, activation: :relu)
  end

  defp se_block(x, channels, reduction \\ 8) do
    # Squeeze-and-Excitation Block
    # Re-calibrates channel importance based on global context
    
    squeeze_channels = div(channels, reduction) |> max(4)
    
    # Global Average Pooling (Squeeze)
    # keep_axes: true ensures we get [Batch, 1, Channels] for broadcasting
    se_weights = 
      x
      |> Axon.global_avg_pool(keep_axes: true)
      |> Axon.dense(squeeze_channels, activation: :relu)
      |> Axon.dense(channels, activation: :sigmoid)
    
    # Scale (Excitation)
    Axon.multiply(x, se_weights)
  end

  defp mixed_pooling(x) do
    # Concatenate Global Max and Global Avg pooling
    Axon.concatenate([
      Axon.global_max_pool(x),
      Axon.global_avg_pool(x)
    ], axis: -1)
  end
end
