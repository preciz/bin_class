defmodule BinClass.Vectorizer do
  alias Tokenizers.{Tokenizer, Encoding}

  def build(tokenizer, string, vector_length) when is_binary(string) do
    {:ok, encoding} = Tokenizer.encode(tokenizer, string)

    encoding
    |> Encoding.truncate(vector_length)
    |> Encoding.pad(vector_length)
    |> Encoding.get_ids()
  end
end
