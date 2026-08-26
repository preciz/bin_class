defmodule BinClass.Vectorizer do
  @moduledoc """
  Encodes text into fixed-length token ID vectors with explicit padding semantics.
  """

  alias Tokenizers.{Encoding, Tokenizer}

  @doc """
  Encodes one string into exactly `vector_length` token IDs.

  Tokens beyond the vector length are truncated. Shorter encodings are padded
  with `pad_token_id`, which should come from `BinClass.Tokenizer.metadata/1`.
  """
  def build(tokenizer, string, vector_length, pad_token_id)
      when is_binary(string) and is_integer(pad_token_id) and pad_token_id >= 0 do
    tokenizer
    |> encode!(string)
    |> Encoding.truncate(vector_length)
    |> Encoding.pad(vector_length, pad_id: pad_token_id, pad_token: "[PAD]")
    |> Encoding.get_ids()
  end

  @doc """
  Encodes one string and also returns its active token count.

  Returns `{token_ids, active_token_count}`. The token IDs are truncated and
  padded to `vector_length`; the count is calculated after truncation and
  excludes IDs equal to `pad_token_id`.
  """
  def build_with_active_token_count(tokenizer, string, vector_length, pad_token_id)
      when is_binary(string) and is_integer(pad_token_id) and pad_token_id >= 0 do
    encoding =
      tokenizer
      |> encode!(string)
      |> Encoding.truncate(vector_length)

    active_token_count =
      encoding
      |> Encoding.get_ids()
      |> BinClass.Padding.active_token_count(pad_token_id)

    ids =
      encoding
      |> Encoding.pad(vector_length, pad_id: pad_token_id, pad_token: "[PAD]")
      |> Encoding.get_ids()

    {ids, active_token_count}
  end

  defp encode!(tokenizer, string) do
    {:ok, encoding} = Tokenizer.encode(tokenizer, string)
    encoding
  end
end
