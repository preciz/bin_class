defmodule BinClass.Padding do
  @moduledoc false

  def active_token_count(token_ids, pad_token_id)
      when is_list(token_ids) and is_integer(pad_token_id) do
    Enum.count(token_ids, &(&1 != pad_token_id))
  end

  def active_token_mask(token_ids, pad_token_id) do
    Nx.not_equal(token_ids, pad_token_id)
  end

  def active_token_counts(token_ids, pad_token_id) do
    token_ids
    |> active_token_mask(pad_token_id)
    |> Nx.sum(axes: [-1])
  end
end
