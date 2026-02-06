defmodule BinClass.TokenizerTest do
  use ExUnit.Case
  alias BinClass.Tokenizer

  test "trains a tokenizer and converts to json" do
    data = ["hello world", "this is a test", "another example"]
    tokenizer = Tokenizer.train(data, vocab_size: 100)

    assert tokenizer

    # Check if we can encode something
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, "hello test")
    assert Tokenizers.Encoding.get_ids(encoding) |> length() > 0

    json = Tokenizer.to_json(tokenizer)
    assert is_binary(json)
    assert String.length(json) > 0

    # Check if we can reload from json
    {:ok, reloaded_tokenizer} = Tokenizers.Tokenizer.from_buffer(json)
    {:ok, encoding2} = Tokenizers.Tokenizer.encode(reloaded_tokenizer, "hello test")
    assert Tokenizers.Encoding.get_ids(encoding) == Tokenizers.Encoding.get_ids(encoding2)
  end

  test "vocab size defaults" do
    assert Tokenizer.vocab_size() == 30_000
  end
end
