defmodule BinClass.Tokenizer do
  @moduledoc """
  Trains tokenizers and manages the metadata required by classifier artifacts.
  """

  alias BinClass.Options
  alias Tokenizers.{Model, Normalizer, PreTokenizer, Tokenizer, Trainer}

  @default_vocab_size 30_000
  @default_normalizer :nfkc_lowercase
  @pad_token "[PAD]"
  @unk_token "[UNK]"

  @doc """
  Returns the vocabulary size and required special-token IDs for a tokenizer.
  """
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

  @doc false
  def validate_metadata!(tokenizer, expected) when is_map(expected) do
    actual = metadata(tokenizer)

    Enum.each([:vocab_size, :pad_token_id, :unk_token_id], fn key ->
      expected_value = Map.fetch!(expected, key)
      actual_value = Map.fetch!(actual, key)

      if expected_value != actual_value do
        raise ArgumentError,
              "stored tokenizer metadata #{key} is #{inspect(expected_value)}, " <>
                "but tokenizer JSON contains #{inspect(actual_value)}"
      end
    end)

    actual
  end

  @doc """
  Trains a BPE tokenizer from an enumerable of strings.

  ## Options

    * `:vocab_size` - Positive maximum vocabulary size. It must be at least
      `5` so all required special tokens can be included. Defaults to
      `#{@default_vocab_size}`.
    * `:normalizer` - Text normalizer to apply before tokenization. Accepts
      `:nfkc_lowercase` (Unicode NFKC + lowercasing, default), `:nfkc`,
      `:lowercase`, `:none`, `nil`, or a custom `Tokenizers.Normalizer` struct.
      Defaults to `:nfkc_lowercase`.

  The tokenizer always includes `[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, and
  `[MASK]` as special tokens.
  """
  def train(data, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        vocab_size: @default_vocab_size,
        normalizer: @default_normalizer
      )

    vocab_size =
      opts |> Keyword.fetch!(:vocab_size) |> Options.validate_positive_integer!(:vocab_size)

    normalizer = validate_normalizer!(Keyword.fetch!(opts, :normalizer))

    if vocab_size < 5 do
      raise ArgumentError,
            ":vocab_size must be at least 5 to include the required special tokens, " <>
              "got: #{vocab_size}"
    end

    {:ok, model} = Model.BPE.init(%{}, [], unk_token: "[UNK]")
    {:ok, tokenizer} = Tokenizer.init(model)

    tokenizer = apply_normalizer(tokenizer, normalizer)

    pre_tokenizer =
      PreTokenizer.sequence([
        PreTokenizer.split(
          "
",
          :removed
        ),
        PreTokenizer.whitespace()
      ])

    tokenizer = Tokenizer.set_pre_tokenizer(tokenizer, pre_tokenizer)

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

  @doc false
  def to_json(tokenizer) do
    BinClass.Tmp.with_tmp_dir(fn dir ->
      tmp_path = Path.join(dir, "tokenizer.json")
      Tokenizer.save(tokenizer, tmp_path)
      File.read!(tmp_path)
    end)
  end

  defp apply_normalizer(tokenizer, :none), do: tokenizer
  defp apply_normalizer(tokenizer, nil), do: tokenizer

  defp apply_normalizer(tokenizer, :nfkc_lowercase) do
    normalizer = Normalizer.sequence([Normalizer.nfkc(), Normalizer.lowercase()])
    Tokenizer.set_normalizer(tokenizer, normalizer)
  end

  defp apply_normalizer(tokenizer, :nfkc) do
    Tokenizer.set_normalizer(tokenizer, Normalizer.nfkc())
  end

  defp apply_normalizer(tokenizer, :lowercase) do
    Tokenizer.set_normalizer(tokenizer, Normalizer.lowercase())
  end

  defp apply_normalizer(tokenizer, normalizer) do
    Tokenizer.set_normalizer(tokenizer, normalizer)
  end

  defp validate_normalizer!(normalizer)
       when normalizer in [:nfkc_lowercase, :nfkc, :lowercase, :none, nil],
       do: normalizer

  defp validate_normalizer!(%Tokenizers.Normalizer{} = normalizer), do: normalizer

  defp validate_normalizer!(normalizer) do
    raise ArgumentError,
          ":normalizer must be :nfkc_lowercase, :nfkc, :lowercase, :none, nil, or a " <>
            "Tokenizers.Normalizer struct, got: #{inspect(normalizer)}"
  end

  defp required_token_id!(tokenizer, token) do
    case Tokenizer.token_to_id(tokenizer, token) do
      nil -> raise ArgumentError, "tokenizer is missing required token #{inspect(token)}"
      id -> id
    end
  end
end
