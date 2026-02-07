defmodule BinClass.Trainer do
  require Explorer.Series
  alias BinClass.{Model, Vectorizer, Tokenizer}

  @default_vector_length 256
  @model_version 5

  @doc """
  Trains a binary classifier on the given data stream.

  ## Options

    * `:epochs` - Number of training epochs. Defaults to `10`.
    * `:batch_size` - Batch size for training. Defaults to `32`.
    * `:learning_rate` - Initial learning rate. Defaults to `1.0e-3`.
    * `:decay` - Learning rate decay. Defaults to `1.0e-2`.
    * `:labels` - Mapping of labels. Can be a list or a map. Defaults to `[0, 1]`.
    * `:validation_split` - Fraction of data to use for validation. Defaults to `0.1`.
    * `:patience` - Number of epochs to wait for improvement before early stopping. Defaults to `5`.
    * `:compiler` - The Nx compiler to use. Defaults to `EXLA`.
    * `:model_version` - The architecture version to use. Defaults to `#{@model_version}`.
    * `:tune` - If `true`, performs automatic hyperparameter tuning for learning rate and dropout. Defaults to `false`.
    * `:dropout_rate` - Dropout rate for the model (ignored if `:tune` is `true`). Defaults to `0.2`.
    * `:vector_length` - Fixed sequence length for tokenization. If not provided, it is automatically calculated from data.
    * `:tokenizer_data` - Custom data stream to train the tokenizer. Defaults to the `:text` field of `data_stream`.
  """
  def train(data_stream, opts \\ []) do
    tokenizer_data_stream = Keyword.get(opts, :tokenizer_data, Stream.map(data_stream, & &1.text))
    epochs = Keyword.get(opts, :epochs, 10)
    batch_size = Keyword.get(opts, :batch_size, 32)
    learning_rate = Keyword.get(opts, :learning_rate, 1.0e-3)
    decay = Keyword.get(opts, :decay, 1.0e-2)
    labels = Keyword.get(opts, :labels, [0, 1])
    validation_split = Keyword.get(opts, :validation_split, 0.1)
    patience = Keyword.get(opts, :patience, 5)
    compiler = Keyword.get(opts, :compiler, EXLA)
    model_version = Keyword.get(opts, :model_version, @model_version)
    tune? = Keyword.get(opts, :tune, false)

    tokenizer = Tokenizer.train(tokenizer_data_stream)
    vocab_size = Tokenizer.vocab_size()

    vector_length =
      case Keyword.get(opts, :vector_length) do
        nil ->
          lengths =
            data_stream
            |> Stream.map(fn %{text: text} ->
              {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)
              length(Tokenizers.Encoding.get_ids(encoding))
            end)
            |> Enum.to_list()

          percentile_90(lengths)

        val ->
          val
      end

    data_stream = balance_data(data_stream)
    df = Explorer.DataFrame.new(data_stream) |> Explorer.DataFrame.shuffle()
    series = Explorer.DataFrame.to_series(df)

    count = Explorer.Series.count(series["text"])

    data =
      series["text"]
      |> Explorer.Series.to_list()
      |> Task.async_stream(fn text -> Vectorizer.build(tokenizer, text, vector_length) end,
        timeout: :infinity
      )
      |> Enum.flat_map(fn {:ok, vector} -> vector end)

    batch_count = div(count, batch_size)
    train_batches_count = floor(batch_count * (1.0 - validation_split))

    {train_data, test_data} =
      data
      |> Nx.tensor(type: :u16)
      |> Nx.reshape({count, vector_length})
      |> Nx.to_batched(batch_size, leftover: :discard)
      |> Enum.split(train_batches_count)

    {train_labels, test_labels} =
      series["label"]
      |> Explorer.Series.to_tensor()
      |> Nx.new_axis(-1)
      |> Nx.equal(Nx.tensor([0, 1]))
      |> Nx.to_batched(batch_size, leftover: :discard)
      |> Enum.split(train_batches_count)

    {best_lr, best_dropout} =
      if tune? do
        tune_hyperparameters(
          train_data,
          train_labels,
          test_data,
          test_labels,
          model_version,
          vocab_size,
          compiler
        )
      else
        {learning_rate, Keyword.get(opts, :dropout_rate, 0.2)}
      end

    build_model_fn = fn ->
      Model.build(model_version, vocab_size, dropout_rate: best_dropout)
    end

    optimizer = Polaris.Optimizers.adamw(learning_rate: best_lr, decay: decay)

    train_model = Axon.Loop.trainer(build_model_fn.(), :categorical_cross_entropy, optimizer)

    best_checkpoint =
      checkpointed_loop_run(%{
        train_model: train_model,
        build_model_fn: build_model_fn,
        train_data: Stream.zip(train_data, train_labels),
        test_data: Stream.zip(test_data, test_labels),
        epochs: epochs,
        patience: patience,
        compiler: compiler
      })

    %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: best_checkpoint.checkpoint.step_state.model_state,
      accuracy: best_checkpoint.accuracy,
      epoch: best_checkpoint.epoch,
      vector_length: vector_length,
      vocab_size: vocab_size,
      labels: labels,
      model_version: model_version,
      learning_rate: best_lr,
      dropout_rate: best_dropout
    }
  end

  defp tune_hyperparameters(
         train_data,
         train_labels,
         test_data,
         test_labels,
         model_version,
         vocab_size,
         compiler
       ) do
    learning_rates = [1.0e-2, 1.0e-3, 5.0e-4, 1.0e-4]
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    tune_limit = max(1, floor(Enum.count(train_data) * 0.2))

    tune_train_data =
      Stream.zip(Enum.take(train_data, tune_limit), Enum.take(train_labels, tune_limit))

    tune_test_data = Stream.zip(test_data, test_labels)

    results =
      for lr <- learning_rates, dr <- dropout_rates do
        build_fn = fn -> Model.build(model_version, vocab_size, dropout_rate: dr) end
        optimizer = Polaris.Optimizers.adamw(learning_rate: lr, decay: 1.0e-2)

        model = build_fn.()
        trainer = Axon.Loop.trainer(model, :categorical_cross_entropy, optimizer)

        final_state =
          trainer
          |> Axon.Loop.validate(model, tune_test_data)
          |> Axon.Loop.run(tune_train_data, %{},
            epochs: 2,
            compiler: compiler,
            garbage_collect: true
          )

        accuracy = calculate_accuracy(model, final_state, tune_test_data, compiler)

        %{lr: lr, dr: dr, accuracy: accuracy}
      end

    best = Enum.max_by(results, & &1.accuracy)

    {best.lr, best.dr}
  end

  defp percentile_90([]), do: @default_vector_length

  defp percentile_90(list) do
    sorted = Enum.sort(list)
    count = length(sorted)
    index = floor(count * 0.90)
    Enum.at(sorted, index) |> max(5)
  end

  defp balance_data(data_stream) do
    data = Enum.to_list(data_stream)
    {class_0, class_1} = Enum.split_with(data, &(&1.label == 0))

    c0 = length(class_0)
    c1 = length(class_1)

    cond do
      c0 > c1 and c1 > 0 ->
        class_1_oversampled = Stream.cycle(class_1) |> Enum.take(c0)
        class_0 ++ class_1_oversampled

      c1 > c0 and c0 > 0 ->
        class_0_oversampled = Stream.cycle(class_0) |> Enum.take(c1)
        class_1 ++ class_0_oversampled

      true ->
        data
    end
  end

  defp checkpointed_loop_run(
         %{
           train_model: train_model,
           build_model_fn: build_model_fn,
           train_data: train_data,
           test_data: test_data,
           epochs: epochs,
           patience: patience,
           compiler: compiler
         } = map,
         opts \\ []
       ) do
    init_state = Map.get(map, :init_state, Axon.ModelState.empty())

    opts =
      opts
      |> Keyword.put(:epochs, epochs)
      |> Keyword.merge(garbage_collect: true, compiler: compiler)

    BinClass.Tmp.with_tmp_dir(fn dir ->
      checkpoints_dir = Path.join(dir, "checkpoints")

      train_model
      |> Axon.Loop.validate(build_model_fn.(), test_data)
      |> Axon.Loop.early_stop("validation_loss", patience: patience)
      |> Axon.Loop.checkpoint(
        event: :epoch_completed,
        path: checkpoints_dir,
        file_pattern: &"#{&1.epoch}.ckpt"
      )
      |> Axon.Loop.run(train_data, init_state, opts)

      model = build_model_fn.()

      Path.wildcard(Path.join(checkpoints_dir, "*.ckpt"))
      |> Enum.map(fn checkpoint_path ->
        epoch_str = Path.basename(checkpoint_path, ".ckpt")
        epoch = String.to_integer(epoch_str)
        checkpoint = File.read!(checkpoint_path) |> Axon.Loop.deserialize_state()

        accuracy =
          calculate_accuracy(model, checkpoint.step_state.model_state, test_data, compiler)

        %{epoch: epoch, accuracy: accuracy, checkpoint: checkpoint}
      end)
      |> Enum.max_by(& &1.accuracy)
    end)
  end

  defp calculate_accuracy(model, trained_model_state, test_data, compiler) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "accuracy")
    |> Axon.Loop.run(test_data, trained_model_state, compiler: compiler)
    |> Map.get(0)
    |> Map.get("accuracy")
    |> Nx.to_number()
  end
end