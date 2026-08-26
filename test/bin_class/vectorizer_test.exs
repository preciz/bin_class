defmodule BinClass.VectorizerTest do
  use ExUnit.Case
  alias BinClass.{Vectorizer, Tokenizer}

  setup do
    data = ["hello world", "this is a test", "another example"]
    tokenizer = Tokenizer.train(data, vocab_size: 100)
    %{tokenizer: tokenizer, metadata: Tokenizer.metadata(tokenizer)}
  end

  test "vectorizes string", %{tokenizer: tokenizer, metadata: metadata} do
    string = "hello world"
    vector_length = 10

    ids = Vectorizer.build(tokenizer, string, vector_length, metadata.pad_token_id)

    assert is_list(ids)
    assert length(ids) == vector_length
    assert Enum.all?(ids, &is_integer/1)
  end

  test "pads shorter strings", %{tokenizer: tokenizer, metadata: metadata} do
    string = "hello"
    vector_length = 20

    ids = Vectorizer.build(tokenizer, string, vector_length, metadata.pad_token_id)
    assert length(ids) == vector_length
  end

  test "truncates longer strings", %{tokenizer: tokenizer, metadata: metadata} do
    string = "hello world this is a test another example"
    vector_length = 2

    ids = Vectorizer.build(tokenizer, string, vector_length, metadata.pad_token_id)
    assert length(ids) == vector_length
  end

  test "does not conflate unknown tokens with padding" do
    tokenizer = Tokenizer.train(["hello"], vocab_size: 100)
    metadata = Tokenizer.metadata(tokenizer)
    {:ok, raw_encoding} = Tokenizers.Tokenizer.encode(tokenizer, "hello ☃")
    raw_ids = Tokenizers.Encoding.get_ids(raw_encoding)

    {ids, active_length} =
      Vectorizer.build_with_length(tokenizer, "hello ☃", 8, metadata.pad_token_id)

    assert metadata.unk_token_id in raw_ids
    assert active_length == length(raw_ids)
    assert Enum.take(ids, length(raw_ids)) == raw_ids

    assert Enum.drop(ids, length(raw_ids)) ==
             List.duplicate(metadata.pad_token_id, 8 - length(raw_ids))

    assert Enum.count(ids, &(&1 != metadata.pad_token_id)) == length(raw_ids)
  end

  test "reports the truncated active length", %{tokenizer: tokenizer, metadata: metadata} do
    {ids, active_length} =
      Vectorizer.build_with_length(
        tokenizer,
        "hello world this is a test another example",
        2,
        metadata.pad_token_id
      )

    assert length(ids) == 2
    assert active_length == 2
  end
end
