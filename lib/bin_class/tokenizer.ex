defmodule BinClass.Tokenizer do
  alias Tokenizers.{Tokenizer, Trainer, PreTokenizer, Model}

  @vocab_size 30_000

  def vocab_size, do: @vocab_size

  def train(data, opts \\ []) do
    vocab_size = Keyword.get(opts, :vocab_size, @vocab_size)
    data = Enum.join(data, "
")

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
        File.write!(tmp_path, data)

        Tokenizer.train_from_files(tokenizer, [tmp_path], trainer: trainer)
      end)

    tokenizer
  end

  def to_json(tokenizer) do
    BinClass.Tmp.with_tmp_dir(fn dir ->
      tmp_path = Path.join(dir, "tokenizer.json")
      Tokenizer.save(tokenizer, tmp_path)
      File.read!(tmp_path)
    end)
  end
end
