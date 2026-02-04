defmodule BinClass.VectorizerTest do
  use ExUnit.Case
  alias BinClass.{Vectorizer, Tokenizer}

  setup do
    data = ["hello world", "this is a test", "another example"]
    tokenizer = Tokenizer.train(data, vocab_size: 100)
    %{tokenizer: tokenizer}
  end

  test "vectorizes string", %{tokenizer: tokenizer} do
    string = "hello world"
    vector_length = 10

    ids = Vectorizer.build(tokenizer, string, vector_length)

    assert is_list(ids)
    assert length(ids) == vector_length
    assert Enum.all?(ids, &is_integer/1)
  end

  test "pads shorter strings", %{tokenizer: tokenizer} do
    string = "hello"
    vector_length = 20

    ids = Vectorizer.build(tokenizer, string, vector_length)
    assert length(ids) == vector_length
  end

  test "truncates longer strings", %{tokenizer: tokenizer} do
    string = "hello world this is a test another example"
    vector_length = 2

    ids = Vectorizer.build(tokenizer, string, vector_length)
    assert length(ids) == vector_length
  end
end
