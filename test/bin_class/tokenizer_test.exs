defmodule BinClass.TokenizerTest do
  use ExUnit.Case
  alias BinClass.Tokenizer
  alias Tokenizers.Encoding
  alias Tokenizers.Model.BPE
  alias Tokenizers.Tokenizer, as: TokenizerImpl

  test "trains a tokenizer and converts to json" do
    data = ["hello world", "this is a test", "another example"]
    tokenizer = Tokenizer.train(data, vocab_size: 100)

    assert tokenizer

    # Check if we can encode something
    {:ok, encoding} = TokenizerImpl.encode(tokenizer, "hello test")
    assert Encoding.get_ids(encoding) |> length() > 0

    json = Tokenizer.to_json(tokenizer)
    assert is_binary(json)
    assert String.length(json) > 0

    # Check if we can reload from json
    {:ok, reloaded_tokenizer} = TokenizerImpl.from_buffer(json)
    {:ok, encoding2} = TokenizerImpl.encode(reloaded_tokenizer, "hello test")
    assert Encoding.get_ids(encoding) == Encoding.get_ids(encoding2)

    assert Tokenizer.metadata(reloaded_tokenizer) == Tokenizer.metadata(tokenizer)
  end

  test "reports actual vocabulary and required special-token IDs" do
    tokenizer = Tokenizer.train(["hello world"], vocab_size: 100)

    assert %{vocab_size: vocab_size, pad_token_id: pad_token_id, unk_token_id: unk_token_id} =
             Tokenizer.metadata(tokenizer)

    assert vocab_size == TokenizerImpl.get_vocab_size(tokenizer)
    assert vocab_size < 100
    assert pad_token_id == TokenizerImpl.token_to_id(tokenizer, "[PAD]")
    assert unk_token_id == TokenizerImpl.token_to_id(tokenizer, "[UNK]")
    assert pad_token_id != unk_token_id
  end

  test "rejects unknown training options" do
    assert_raise ArgumentError, ~r/unknown keys.*:not_an_option/s, fn ->
      Tokenizer.train(["hello world"], not_an_option: true)
    end
  end

  test "rejects invalid vocabulary sizes" do
    assert Tokenizer.train(["hello world"], vocab_size: 5)

    for vocab_size <- [0, -1, :many] do
      assert_raise ArgumentError, ~r/:vocab_size must be a positive integer/, fn ->
        Tokenizer.train(["hello world"], vocab_size: vocab_size)
      end
    end

    for vocab_size <- [1, 4] do
      assert_raise ArgumentError, ~r/:vocab_size must be at least 5/, fn ->
        Tokenizer.train(["hello world"], vocab_size: vocab_size)
      end
    end
  end

  test "rejects invalid training rows and tokenizers missing required special tokens" do
    assert_raise ArgumentError, ~r/training data must contain strings/, fn ->
      Tokenizer.train(["valid", :invalid])
    end

    {:ok, model} = BPE.init(%{}, [], unk_token: "[UNK]")
    {:ok, tokenizer} = TokenizerImpl.init(model)

    assert_raise ArgumentError, ~r/missing required token "\[PAD\]"/, fn ->
      Tokenizer.metadata(tokenizer)
    end
  end

  test "configures pre-tokenizer sequence with newline split and whitespace" do
    tokenizer = Tokenizer.train(["hello\nworld foo bar"], vocab_size: 100)
    json = Tokenizer.to_json(tokenizer)
    {:ok, decoded} = JSON.decode(json)

    assert %{
             "type" => "Sequence",
             "pretokenizers" => [
               %{"type" => "Split", "behavior" => "Removed"},
               %{"type" => "Whitespace"}
             ]
           } = decoded["pre_tokenizer"]
  end
end
