defmodule BinClass.ModelTest do
  use ExUnit.Case
  alias BinClass.Model

  test "builds an axon model" do
    vocab_size = 100
    model = Model.build(vocab_size)

    assert %Axon{} = model

    # Optional: Inspect structure if needed, but just ensuring it returns a struct is a good start.
    # We can check input/output shapes if we compile it, but that might be overkill for unit test
    # unless we want to ensure specific layers exist.
  end
end
