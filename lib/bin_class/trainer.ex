defmodule BinClass.Trainer do
  alias BinClass.{Model, Vectorizer, Tokenizer}

  @default_vector_length 512
  @model_version :conservative_cnn

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
    * `:false_positive_penalty` - Penalty multiplier applied to validation false-positive rate
      when selecting checkpoints and tuned hyperparameters. Defaults to `0.5` for v7 and `0.0`
      for older models.
    * `:calibrate_threshold` - If `true`, calibrates and persists a positive threshold on the
      validation split. Defaults to `true` for v7 and `false` for older models.
    * `:threshold_candidates` - Thresholds to evaluate during calibration. Defaults to
      `0.5..0.9` in `0.05` increments.
    * `:vector_length` - Fixed sequence length for tokenization. Defaults to `512`.
    * `:tokenizer_data` - Custom data stream to train the tokenizer. Defaults to the `:text` field of `data_stream`.
  """
  def train(data_stream, opts \\ []) do
    clear_recompilation_counters()
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

    false_positive_penalty =
      Keyword.get(opts, :false_positive_penalty, default_false_positive_penalty(model_version))

    calibrate_threshold? =
      Keyword.get(opts, :calibrate_threshold, default_threshold_calibration?(model_version))

    threshold_candidates =
      Keyword.get(opts, :threshold_candidates, default_threshold_candidates(model_version))

    tokenizer = Tokenizer.train(tokenizer_data_stream)
    vocab_size = Tokenizer.vocab_size()

    vector_length = Keyword.get(opts, :vector_length, @default_vector_length)

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
          compiler,
          false_positive_penalty
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
        compiler: compiler,
        model_version: model_version,
        false_positive_penalty: false_positive_penalty,
        calibrate_threshold?: calibrate_threshold?,
        threshold_candidates: threshold_candidates
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
      dropout_rate: best_dropout,
      decision_policy: best_checkpoint.decision_policy
    }
  end

  defp tune_hyperparameters(
         train_data,
         train_labels,
         test_data,
         test_labels,
         model_version,
         vocab_size,
         compiler,
         false_positive_penalty
       ) do
    learning_rates = [1.0e-2, 1.0e-3, 5.0e-4, 1.0e-4]
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    tune_limit = max(1, floor(Enum.count(train_data) * 0.2))

    tune_train_data =
      Stream.zip(Enum.take(train_data, tune_limit), Enum.take(train_labels, tune_limit))

    tune_test_data = Stream.zip(test_data, test_labels)

    results =
      for lr <- learning_rates, dr <- dropout_rates do
        clear_recompilation_counters()
        build_fn = fn -> Model.build(model_version, vocab_size, dropout_rate: dr) end
        optimizer = Polaris.Optimizers.adamw(learning_rate: lr, decay: 1.0e-2)

        model = build_fn.()
        trainer = Axon.Loop.trainer(model, :categorical_cross_entropy, optimizer)

        final_state =
          trainer
          |> Axon.Loop.validate(model, tune_test_data)
          |> Axon.Loop.run(tune_train_data, Axon.ModelState.empty(),
            epochs: 2,
            compiler: compiler,
            garbage_collect: true
          )

        {_, predict_fn} = Axon.build(model, compiler: compiler)

        metrics =
          calculate_validation_metrics(predict_fn, final_state, tune_test_data, model_version)

        score = validation_score(metrics, false_positive_penalty)

        %{lr: lr, dr: dr, accuracy: metrics.accuracy, score: score}
      end

    best = Enum.max_by(results, & &1.score)

    {best.lr, best.dr}
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
           compiler: compiler,
           model_version: model_version,
           false_positive_penalty: false_positive_penalty,
           calibrate_threshold?: calibrate_threshold?,
           threshold_candidates: threshold_candidates
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
      {_, predict_fn} = Axon.build(model, compiler: compiler)

      best_checkpoint =
        Path.wildcard(Path.join(checkpoints_dir, "*.ckpt"))
        |> Enum.map(fn checkpoint_path ->
          epoch_str = Path.basename(checkpoint_path, ".ckpt")
          epoch = String.to_integer(epoch_str)
          checkpoint = File.read!(checkpoint_path) |> Axon.Loop.deserialize_state()

          metrics =
            calculate_validation_metrics(
              predict_fn,
              checkpoint.step_state.model_state,
              test_data,
              model_version
            )

          %{
            epoch: epoch,
            accuracy: metrics.accuracy,
            validation_score: validation_score(metrics, false_positive_penalty),
            checkpoint: checkpoint
          }
        end)
        |> Enum.max_by(& &1.validation_score)

      decision_policy =
        if calibrate_threshold? and BinClass.Serving.decision_policy?(model_version) do
          calibrate_decision_policy(
            predict_fn,
            best_checkpoint.checkpoint.step_state.model_state,
            test_data,
            model_version,
            false_positive_penalty,
            threshold_candidates
          )
        end

      Map.put(best_checkpoint, :decision_policy, decision_policy)
    end)
  end

  defp calculate_validation_metrics(
         predict_fn,
         trained_model_state,
         test_data,
         model_version,
         decision_policy \\ nil
       ) do
    {correct, total, false_positives, true_negatives} =
      Enum.reduce(test_data, {0, 0, 0, 0}, fn {inputs, labels},
                                              {correct_acc, count_acc, fp_acc, tn_acc} ->
        preds = predict_fn.(trained_model_state, inputs)

        pred_labels = prediction_labels(preds, inputs, model_version, decision_policy)
        true_labels = Nx.argmax(labels, axis: -1)

        correct = Nx.equal(pred_labels, true_labels) |> Nx.sum() |> Nx.to_number()
        false_positives = false_positive_count(pred_labels, true_labels)
        true_negatives = true_negative_count(pred_labels, true_labels)
        batch_size = Nx.shape(inputs) |> elem(0)

        {
          correct_acc + correct,
          count_acc + batch_size,
          fp_acc + false_positives,
          tn_acc + true_negatives
        }
      end)

    false_positive_denominator = false_positives + true_negatives

    false_positive_rate =
      if false_positive_denominator == 0 do
        0.0
      else
        false_positives / false_positive_denominator
      end

    %{
      accuracy: correct / total,
      false_positive_rate: false_positive_rate,
      false_positives: false_positives,
      true_negatives: true_negatives
    }
  end

  defp prediction_labels(preds, inputs, model_version)
       when model_version in [7, :conservative_cnn] do
    policy = BinClass.Serving.decision_policy(model_version)

    prediction_labels_with_policy(preds, inputs, policy)
  end

  defp prediction_labels(preds, _inputs, _model_version) do
    Nx.argmax(preds, axis: -1)
  end

  defp prediction_labels(preds, inputs, _model_version, decision_policy)
       when is_map(decision_policy) do
    prediction_labels_with_policy(preds, inputs, decision_policy)
  end

  defp prediction_labels(preds, inputs, model_version, _decision_policy) do
    prediction_labels(preds, inputs, model_version)
  end

  defp prediction_labels_with_policy(preds, inputs, policy) do
    batch_size = Nx.shape(inputs) |> elem(0)

    positive_probs =
      preds
      |> Nx.slice([0, 1], [batch_size, 1])
      |> Nx.squeeze(axes: [1])

    active_lengths =
      inputs
      |> Nx.not_equal(0)
      |> Nx.sum(axes: [-1])

    positive? =
      positive_probs
      |> Nx.greater_equal(policy.positive_threshold)
      |> Nx.logical_and(Nx.greater_equal(active_lengths, policy.min_positive_tokens))

    Nx.select(
      positive?,
      Nx.broadcast(1, Nx.shape(positive?)),
      Nx.broadcast(0, Nx.shape(positive?))
    )
  end

  defp false_positive_count(pred_labels, true_labels) do
    pred_labels
    |> Nx.equal(1)
    |> Nx.logical_and(Nx.equal(true_labels, 0))
    |> Nx.sum()
    |> Nx.to_number()
  end

  defp true_negative_count(pred_labels, true_labels) do
    pred_labels
    |> Nx.equal(0)
    |> Nx.logical_and(Nx.equal(true_labels, 0))
    |> Nx.sum()
    |> Nx.to_number()
  end

  defp validation_score(metrics, false_positive_penalty) do
    metrics.accuracy - false_positive_penalty * metrics.false_positive_rate
  end

  defp default_false_positive_penalty(model_version)
       when model_version in [7, :conservative_cnn],
       do: 0.5

  defp default_false_positive_penalty(_model_version), do: 0.0

  defp calibrate_decision_policy(
         predict_fn,
         trained_model_state,
         test_data,
         model_version,
         false_positive_penalty,
         threshold_candidates
       ) do
    base_policy = BinClass.Serving.decision_policy(model_version)

    threshold_candidates
    |> Enum.map(fn threshold ->
      policy = %{base_policy | positive_threshold: threshold}

      metrics =
        calculate_validation_metrics(
          predict_fn,
          trained_model_state,
          test_data,
          model_version,
          policy
        )

      %{
        policy: policy,
        score: validation_score(metrics, false_positive_penalty),
        accuracy: metrics.accuracy,
        false_positive_rate: metrics.false_positive_rate
      }
    end)
    |> Enum.max_by(fn result ->
      {result.score, result.accuracy, result.policy.positive_threshold}
    end)
    |> then(fn result ->
      result.policy
      |> Map.put(:validation_score, result.score)
      |> Map.put(:validation_accuracy, result.accuracy)
      |> Map.put(:validation_false_positive_rate, result.false_positive_rate)
    end)
  end

  defp default_threshold_calibration?(model_version)
       when model_version in [7, :conservative_cnn],
       do: true

  defp default_threshold_calibration?(_model_version), do: false

  defp default_threshold_candidates(_model_version) do
    for threshold <- 50..90//5, do: threshold / 100
  end

  defp clear_recompilation_counters do
    if Code.ensure_loaded?(EXLA.Defn.LockedCache) and
         :ets.info(EXLA.Defn.LockedCache) != :undefined do
      :ets.match_delete(EXLA.Defn.LockedCache, {{:counter, :_}, :_})
    end
  rescue
    _ -> :ok
  end
end
