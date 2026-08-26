defmodule BinClass.VectorizationPropertyTest do
  use ExUnit.Case
  use ExUnitProperties

  alias BinClass.{Padding, Tokenizer, Vectorizer}
  alias Tokenizers.Encoding
  alias Tokenizers.Tokenizer, as: TokenizerImpl

  setup_all do
    tokenizer =
      Tokenizer.train([
        "alpha beta gamma delta",
        "Unicode text: café ☃ λ",
        "[PAD] is a literal special token"
      ])

    %{tokenizer: tokenizer, metadata: Tokenizer.metadata(tokenizer)}
  end

  property "list and tensor padding operations use the same predicate" do
    check all(
            token_ids <- list_of(integer(0..31), min_length: 1, max_length: 80),
            pad_token_id <- integer(0..31),
            max_runs: 200
          ) do
      expected_mask = Enum.map(token_ids, &if(&1 == pad_token_id, do: 0, else: 1))
      expected_count = Enum.sum(expected_mask)
      tensor = Nx.tensor([token_ids])

      assert Padding.active_token_count(token_ids, pad_token_id) == expected_count
      assert Padding.active_token_mask(tensor, pad_token_id) |> Nx.to_flat_list() == expected_mask

      assert Padding.active_token_counts(tensor, pad_token_id) |> Nx.to_flat_list() ==
               [expected_count]
    end
  end

  property "vectorization pads or truncates while reporting the exact active count", %{
    tokenizer: tokenizer,
    metadata: metadata
  } do
    check all(
            text <- text(),
            vector_length <- integer(1..64),
            max_runs: 100
          ) do
      {:ok, encoding} = TokenizerImpl.encode(tokenizer, text)

      raw_ids =
        encoding
        |> Encoding.truncate(vector_length)
        |> Encoding.get_ids()

      {ids, active_token_count} =
        Vectorizer.build_with_active_token_count(
          tokenizer,
          text,
          vector_length,
          metadata.pad_token_id
        )

      assert length(ids) == vector_length
      assert Enum.take(ids, length(raw_ids)) == raw_ids

      assert Enum.drop(ids, length(raw_ids)) ==
               List.duplicate(metadata.pad_token_id, vector_length - length(raw_ids))

      assert active_token_count == Enum.count(raw_ids, &(&1 != metadata.pad_token_id))
      assert active_token_count == Enum.count(ids, &(&1 != metadata.pad_token_id))
    end
  end

  defp text do
    one_of([
      string(:utf8, max_length: 80),
      list_of(member_of(["alpha", "βeta", "☃", "[PAD]", "unknown", "\n"]), max_length: 20)
      |> map(&Enum.join(&1, " "))
    ])
  end
end
