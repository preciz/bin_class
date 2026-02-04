defmodule BinClassTest do
  use ExUnit.Case
  doctest BinClass

  test "modules are loaded" do
    assert Code.ensure_loaded?(BinClass.Trainer)
    assert Code.ensure_loaded?(BinClass.Serving)
    assert Code.ensure_loaded?(BinClass.Model)
    assert Code.ensure_loaded?(BinClass.Tokenizer)
  end
end
