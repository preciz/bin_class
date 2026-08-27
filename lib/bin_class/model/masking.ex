defmodule BinClass.Model.Masking do
  @moduledoc false

  defstruct [:token_mask]

  def new(token_ids, pad_token_id) do
    token_mask =
      Axon.layer(
        fn token_ids, _opts -> BinClass.Padding.active_token_mask(token_ids, pad_token_id) end,
        [token_ids]
      )

    %__MODULE__{token_mask: token_mask}
  end

  def valid_convolution(%__MODULE__{token_mask: token_mask}, kernel_size)
      when is_integer(kernel_size) and kernel_size > 0 do
    convolved_mask =
      Axon.layer(
        fn token_mask, _opts -> valid_convolution_mask(token_mask, kernel_size) end,
        [token_mask]
      )

    %__MODULE__{token_mask: convolved_mask}
  end

  def mask_sequence(sequence, %__MODULE__{token_mask: token_mask}) do
    Axon.layer(
      fn sequence, token_mask, _opts ->
        mask =
          token_mask
          |> Nx.new_axis(-1)
          |> Nx.broadcast(Nx.shape(sequence))

        zeros = Nx.broadcast(0, Nx.shape(sequence)) |> Nx.as_type(Nx.type(sequence))
        Nx.select(mask, sequence, zeros)
      end,
      [sequence, token_mask]
    )
  end

  def masked_global_max(sequence, %__MODULE__{token_mask: token_mask}) do
    Axon.layer(
      fn sequence, token_mask, _opts ->
        mask = feature_mask(token_mask, sequence)
        minimum = Nx.broadcast(-1.0e30, Nx.shape(sequence))

        pooled =
          mask
          |> Nx.select(sequence, minimum)
          |> Nx.reduce_max(axes: [1])

        active? =
          token_mask
          |> Nx.any(axes: [1], keep_axes: true)
          |> Nx.broadcast(Nx.shape(pooled))

        zeros = Nx.broadcast(0, Nx.shape(pooled)) |> Nx.as_type(Nx.type(pooled))
        Nx.select(active?, pooled, zeros)
      end,
      [sequence, token_mask]
    )
  end

  def masked_global_avg(sequence, masking, opts \\ [])

  def masked_global_avg(
        sequence,
        %__MODULE__{token_mask: token_mask},
        opts
      ) do
    opts = Keyword.validate!(opts, keep_axes: false)

    keep_axes? =
      opts
      |> Keyword.fetch!(:keep_axes)
      |> BinClass.Options.validate_boolean!(:keep_axes)

    Axon.layer(
      fn sequence, token_mask, _opts ->
        mask = feature_mask(token_mask, sequence)
        mask = Nx.as_type(mask, Nx.type(sequence))

        sums =
          sequence
          |> Nx.multiply(mask)
          |> Nx.sum(axes: [1], keep_axes: keep_axes?)

        counts =
          mask
          |> Nx.sum(axes: [1], keep_axes: keep_axes?)
          |> Nx.max(1)

        Nx.divide(sums, counts)
      end,
      [sequence, token_mask]
    )
  end

  def masked_attentive_pooling(sequence, masking, opts \\ [])

  def masked_attentive_pooling(
        sequence,
        %__MODULE__{token_mask: token_mask},
        opts
      ) do
    opts = Keyword.validate!(opts, heads: 2, hidden_size: 64)
    heads = Keyword.fetch!(opts, :heads)
    hidden_size = Keyword.fetch!(opts, :hidden_size)

    attn_hidden = Axon.dense(sequence, hidden_size, activation: :tanh)
    scores = Axon.dense(attn_hidden, heads)

    weights =
      Axon.layer(
        fn scores, token_mask, _opts ->
          mask =
            token_mask
            |> Nx.new_axis(-1)
            |> Nx.broadcast(Nx.shape(scores))

          minimum = Nx.broadcast(-1.0e30, Nx.shape(scores))
          masked_scores = Nx.select(mask, scores, minimum)

          max_score = Nx.reduce_max(masked_scores, axes: [1], keep_axes: true)
          exp = Nx.exp(Nx.subtract(masked_scores, max_score))
          exp_masked = Nx.select(mask, exp, Nx.broadcast(0.0, Nx.shape(exp)))
          sum_exp = Nx.sum(exp_masked, axes: [1], keep_axes: true) |> Nx.max(1.0e-12)
          Nx.divide(exp_masked, sum_exp)
        end,
        [scores, token_mask]
      )

    Axon.layer(
      fn sequence, weights, _opts ->
        seq_expanded = Nx.new_axis(sequence, 2)
        w_expanded = Nx.new_axis(weights, 3)

        weighted = Nx.multiply(seq_expanded, w_expanded)
        summed = Nx.sum(weighted, axes: [1])
        shape = Nx.shape(summed)
        b = elem(shape, 0)
        h = elem(shape, 1)
        d = elem(shape, 2)
        Nx.reshape(summed, {b, h * d})
      end,
      [sequence, weights]
    )
  end

  def mask_attention_scores(scores, %__MODULE__{token_mask: token_mask}) do
    Axon.layer(
      fn scores, token_mask, _opts ->
        key_mask =
          token_mask
          |> Nx.new_axis(1)
          |> Nx.broadcast(Nx.shape(scores))

        minimum = Nx.broadcast(-1.0e30, Nx.shape(scores))
        Nx.select(key_mask, scores, minimum)
      end,
      [scores, token_mask]
    )
  end

  defp feature_mask(token_mask, sequence) do
    token_mask
    |> Nx.new_axis(-1)
    |> Nx.broadcast(Nx.shape(sequence))
  end

  defp valid_convolution_mask(token_mask, 1), do: token_mask

  defp valid_convolution_mask(token_mask, kernel_size) do
    output_length = Nx.axis_size(token_mask, 1) - kernel_size + 1
    first = Nx.slice_along_axis(token_mask, 0, output_length, axis: 1)

    Enum.reduce(1..(kernel_size - 1), first, fn offset, mask ->
      window = Nx.slice_along_axis(token_mask, offset, output_length, axis: 1)
      Nx.logical_and(mask, window)
    end)
  end
end
