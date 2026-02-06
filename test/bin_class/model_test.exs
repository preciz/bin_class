defmodule BinClass.ModelTest do
  use ExUnit.Case
  alias BinClass.Model

  test "builds an axon model" do
    vocab_size = 100
    model = Model.build(1, vocab_size)

    assert %Axon{} = model

    # Optional: Inspect structure if needed, but just ensuring it returns a struct is a good start.
    # We can check input/output shapes if we compile it, but that might be overkill for unit test
    # unless we want to ensure specific layers exist.
  end

  test "raises on unknown model version" do
    assert_raise ArgumentError, "Unknown model version: 99", fn ->
      Model.build(99, 100)
    end
  end
end
