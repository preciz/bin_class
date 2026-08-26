defmodule BinClass.Vectorizer do
  alias Tokenizers.{Tokenizer, Encoding}

  def build(tokenizer, string, vector_length, pad_token_id)
      when is_binary(string) and is_integer(pad_token_id) and pad_token_id >= 0 do
    tokenizer
    |> encode!(string)
    |> Encoding.truncate(vector_length)
    |> Encoding.pad(vector_length, pad_id: pad_token_id, pad_token: "[PAD]")
    |> Encoding.get_ids()
  end

  def build_with_length(tokenizer, string, vector_length, pad_token_id)
      when is_binary(string) and is_integer(pad_token_id) and pad_token_id >= 0 do
    encoding =
      tokenizer
      |> encode!(string)
      |> Encoding.truncate(vector_length)

    active_length = Encoding.get_length(encoding)

    ids =
      encoding
      |> Encoding.pad(vector_length, pad_id: pad_token_id, pad_token: "[PAD]")
      |> Encoding.get_ids()

    {ids, active_length}
  end

  defp encode!(tokenizer, string) do
    {:ok, encoding} = Tokenizer.encode(tokenizer, string)
    encoding
  end
end
