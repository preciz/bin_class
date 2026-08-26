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

    assert Tokenizer.metadata(reloaded_tokenizer) == Tokenizer.metadata(tokenizer)
  end

  test "reports actual vocabulary and required special-token IDs" do
    tokenizer = Tokenizer.train(["hello world"], vocab_size: 100)

    assert %{vocab_size: vocab_size, pad_token_id: pad_token_id, unk_token_id: unk_token_id} =
             Tokenizer.metadata(tokenizer)

    assert vocab_size == Tokenizers.Tokenizer.get_vocab_size(tokenizer)
    assert vocab_size < 100
    assert pad_token_id == Tokenizers.Tokenizer.token_to_id(tokenizer, "[PAD]")
    assert unk_token_id == Tokenizers.Tokenizer.token_to_id(tokenizer, "[UNK]")
    assert pad_token_id != unk_token_id
  end

  test "rejects unknown training options" do
    assert_raise ArgumentError, ~r/unknown keys.*:not_an_option/s, fn ->
      Tokenizer.train(["hello world"], not_an_option: true)
    end
  end

  test "rejects invalid training rows and tokenizers missing required special tokens" do
    assert_raise ArgumentError, ~r/training data must contain strings/, fn ->
      Tokenizer.train(["valid", :invalid])
    end

    {:ok, model} = Tokenizers.Model.BPE.init(%{}, [], unk_token: "[UNK]")
    {:ok, tokenizer} = Tokenizers.Tokenizer.init(model)

    assert_raise ArgumentError, ~r/missing required token "\[PAD\]"/, fn ->
      Tokenizer.metadata(tokenizer)
    end
  end
end
