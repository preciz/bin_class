defmodule BinClass.Tokenizer do
  alias Tokenizers.{Tokenizer, Trainer, PreTokenizer, Model}

  @default_vocab_size 30_000
  @pad_token "[PAD]"
  @unk_token "[UNK]"

  def metadata(tokenizer) do
    vocab_size = Tokenizer.get_vocab_size(tokenizer)
    pad_token_id = required_token_id!(tokenizer, @pad_token)
    unk_token_id = required_token_id!(tokenizer, @unk_token)

    if pad_token_id == unk_token_id do
      raise ArgumentError,
            "tokenizer padding and unknown tokens must have distinct IDs, both are #{pad_token_id}"
    end

    %{
      vocab_size: vocab_size,
      pad_token_id: pad_token_id,
      unk_token_id: unk_token_id
    }
  end

  def validate_metadata!(tokenizer, expected) when is_map(expected) do
    actual = metadata(tokenizer)

    Enum.each([:vocab_size, :pad_token_id, :unk_token_id], fn key ->
      expected_value = Map.fetch!(expected, key)
      actual_value = Map.fetch!(actual, key)

      unless expected_value == actual_value do
        raise ArgumentError,
              "stored tokenizer metadata #{key} is #{inspect(expected_value)}, " <>
                "but tokenizer JSON contains #{inspect(actual_value)}"
      end
    end)

    actual
  end

  def train(data, opts \\ []) do
    vocab_size = Keyword.get(opts, :vocab_size, @default_vocab_size)

    {:ok, model} = Model.BPE.init(%{}, [], unk_token: "[UNK]")
    {:ok, tokenizer} = Tokenizer.init(model)

    tokenizer =
      tokenizer
      |> Tokenizer.set_pre_tokenizer(
        PreTokenizer.split(
          "
",
          :removed
        )
      )
      |> Tokenizer.set_pre_tokenizer(PreTokenizer.whitespace())

    {:ok, trainer} =
      Trainer.bpe(
        vocab_size: vocab_size,
        special_tokens: ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
      )

    {:ok, tokenizer} =
      BinClass.Tmp.with_tmp_dir(fn dir ->
        tmp_path = Path.join(dir, "temp.txt")
        write_training_data!(tmp_path, data)

        Tokenizer.train_from_files(tokenizer, [tmp_path], trainer: trainer)
      end)

    tokenizer
  end

  defp write_training_data!(path, data) do
    File.open!(path, [:write, :binary], fn file ->
      data
      |> Stream.chunk_every(1_000)
      |> Enum.each(fn texts ->
        IO.binwrite(file, Enum.map(texts, &training_line!/1))
      end)
    end)
  end

  defp training_line!(text) when is_binary(text), do: [text, "\n"]

  defp training_line!(text) do
    raise ArgumentError, "tokenizer training data must contain strings, got: #{inspect(text)}"
  end

  def to_json(tokenizer) do
    BinClass.Tmp.with_tmp_dir(fn dir ->
      tmp_path = Path.join(dir, "tokenizer.json")
      Tokenizer.save(tokenizer, tmp_path)
      File.read!(tmp_path)
    end)
  end

  defp required_token_id!(tokenizer, token) do
    case Tokenizer.token_to_id(tokenizer, token) do
      nil -> raise ArgumentError, "tokenizer is missing required token #{inspect(token)}"
      id -> id
    end
  end
end
