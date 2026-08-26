defmodule BinClass.PaddingTest do
  use ExUnit.Case

  test "uses the same exact non-padding predicate for lists and tensors" do
    token_ids = [7, 3, 8, 3, 9]

    assert BinClass.Padding.active_token_count(token_ids, 3) == 3

    tensor = Nx.tensor([token_ids, [3, 3, 3, 3, 3]])

    assert BinClass.Padding.active_token_mask(tensor, 3) |> Nx.to_list() == [
             [1, 0, 1, 0, 1],
             [0, 0, 0, 0, 0]
           ]

    assert BinClass.Padding.active_token_counts(tensor, 3) |> Nx.to_list() == [3, 0]
  end
end
