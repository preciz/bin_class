defmodule BinClass.Model.Masking do
  @moduledoc false

  defstruct [:token_mask, :active_lengths]

  def new(token_ids, pad_token_id) do
    token_mask =
      Axon.layer(fn token_ids, _opts -> Nx.not_equal(token_ids, pad_token_id) end, [token_ids])

    active_lengths =
      Axon.layer(
        fn token_mask, _opts ->
          token_mask
          |> Nx.sum(axes: [-1], keep_axes: true)
          |> Nx.as_type(:f32)
        end,
        [token_mask]
      )

    %__MODULE__{token_mask: token_mask, active_lengths: active_lengths}
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

  def masked_global_max(sequence, %__MODULE__{active_lengths: active_lengths}) do
    Axon.layer(
      fn sequence, active_lengths, _opts ->
        {mask, active?} = sequence_mask(sequence, active_lengths)
        minimum = Nx.broadcast(-1.0e30, Nx.shape(sequence))

        pooled =
          mask
          |> Nx.select(sequence, minimum)
          |> Nx.reduce_max(axes: [1])

        zeros = Nx.broadcast(0, Nx.shape(pooled)) |> Nx.as_type(Nx.type(pooled))
        Nx.select(active?, pooled, zeros)
      end,
      [sequence, active_lengths]
    )
  end

  def masked_global_avg(sequence, masking, opts \\ [])

  def masked_global_avg(
        sequence,
        %__MODULE__{active_lengths: active_lengths},
        opts
      ) do
    keep_axes? = Keyword.get(opts, :keep_axes, false)

    Axon.layer(
      fn sequence, active_lengths, _opts ->
        {mask, _active?} = sequence_mask(sequence, active_lengths)
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
      [sequence, active_lengths]
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

  defp sequence_mask(sequence, active_lengths) do
    batch_size = Nx.axis_size(sequence, 0)
    sequence_length = Nx.axis_size(sequence, 1)

    active_lengths = Nx.min(active_lengths, sequence_length)

    positions = Nx.iota({1, sequence_length, 1})

    mask =
      positions
      |> Nx.less(Nx.reshape(active_lengths, {batch_size, 1, 1}))
      |> Nx.broadcast(Nx.shape(sequence))

    active? =
      active_lengths
      |> Nx.greater(0)
      |> Nx.broadcast({batch_size, Nx.axis_size(sequence, 2)})

    {mask, active?}
  end
end
