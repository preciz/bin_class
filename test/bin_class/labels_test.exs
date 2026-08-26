defmodule BinClass.LabelsTest do
  use ExUnit.Case, async: true

  alias BinClass.Labels

  test "accepts exactly two distinct binary labels" do
    assert Labels.default() == ["0", "1"]
    assert Labels.validate!(["negative", "positive"]) == ["negative", "positive"]

    assert Labels.validate!(%{0 => "negative", 1 => "positive"}) == %{
             0 => "negative",
             1 => "positive"
           }
  end

  test "rejects labels that cannot be safely persisted" do
    for labels <- [
          [0, 1],
          [:negative, :positive],
          ["duplicate", "duplicate"],
          ["only one"],
          %{0 => "negative"},
          %{0 => "negative", 1 => "positive", 2 => "other"}
        ] do
      assert_raise ArgumentError, ~r/:labels must be.*distinct binaries/, fn ->
        Labels.validate!(labels)
      end
    end
  end
end
