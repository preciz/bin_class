defmodule BinClass.Trainer do
  alias BinClass.{DecisionPolicy, Labels, Model, Options, Vectorizer, Tokenizer}

  @default_vector_length 512
  @architecture :cnn

  @doc """
  Trains a binary classifier on the given data stream.

  ## Options

    * `:epochs` - Positive integer number of training epochs. Defaults to `10`.
    * `:batch_size` - Positive integer batch size for training. Defaults to `32`.
    * `:learning_rate` - Positive initial learning rate. Defaults to `1.0e-3`.
    * `:decay` - Non-negative AdamW weight decay. Defaults to `1.0e-2`.
    * `:labels` - Output label names as a two-element list of distinct binaries or
      `%{0 => negative_binary, 1 => positive_binary}`. Defaults to `["0", "1"]`.
    * `:target_mode` - How training labels are interpreted. `:hard` accepts only `0` and `1`;
      `:soft` accepts positive-class probabilities from `0.0` through `1.0`. Defaults to `:hard`.
    * `:target_threshold` - Boundary used to turn soft targets into binary labels for balancing,
      validation metrics, and threshold calibration. Defaults to `0.5`.
    * `:balance` - Boolean controlling whether to oversample the smaller class in the training
      partition. Validation examples are never oversampled. Soft targets are assigned to a class using
      `:target_threshold`. Defaults to `true`.
    * `:validation_split` - Fraction of full batches to use for validation. Partitions are
      stratified when both target classes can occur in both. Defaults to `0.1`.
    * `:patience` - Non-negative integer number of epochs to wait for improvement before early
      stopping. Defaults to `5`.
    * `:compiler` - A module implementing `Nx.Defn.Compiler`. Defaults to `EXLA`.
    * `:architecture` - The named architecture to use. Defaults to `#{@architecture}`.
    * `:dropout_rate` - Dropout rate from `0.0` up to, but excluding, `1.0`. Defaults to `0.2`.
    * `:decision_policy` - How probabilities become labels. Accepts `:argmax`,
      `{:positive_threshold, threshold}`, `:optimize_threshold`, or
      `{:optimize_threshold, options}`. Threshold optimization options are `:objective`
      (`:accuracy`, `:balanced_accuracy`, or `:f1`) and `:false_positive_cost`.
      Defaults to `:argmax`.
    * `:minimum_input_tokens` - Inputs with fewer active tokens return
      `:insufficient_input` at inference. Defaults to `0`.
    * `:vector_length` - Positive fixed sequence length satisfying the architecture's minimum.
      Defaults to `512`.
    * `:seed` - Optional integer seed for deterministic partition ordering, model initialization,
      and dropout. Defaults to nondeterministic training.
  """
  def train(data_stream, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :epochs,
        :batch_size,
        :learning_rate,
        :decay,
        :labels,
        :target_mode,
        :target_threshold,
        :balance,
        :validation_split,
        :patience,
        :compiler,
        :architecture,
        :dropout_rate,
        :decision_policy,
        :minimum_input_tokens,
        :vector_length,
        :seed
      ])

    target_mode = Keyword.get(opts, :target_mode, :hard)
    target_threshold = Keyword.get(opts, :target_threshold, 0.5)
    balance? = opts |> Keyword.get(:balance, true) |> Options.validate_boolean!(:balance)
    seed = Keyword.get(opts, :seed)

    validate_target_options!(target_mode, target_threshold)
    validate_seed!(seed)

    epochs = Keyword.get(opts, :epochs, 10)
    batch_size = Keyword.get(opts, :batch_size, 32)

    learning_rate =
      opts
      |> Keyword.get(:learning_rate, 1.0e-3)
      |> Options.validate_positive_number!(:learning_rate)

    decay =
      opts
      |> Keyword.get(:decay, 1.0e-2)
      |> Options.validate_non_negative_number!(:decay)

    labels = opts |> Keyword.get(:labels, Labels.default()) |> Labels.validate!()
    validation_split = Keyword.get(opts, :validation_split, 0.1)

    patience =
      opts
      |> Keyword.get(:patience, 5)
      |> Options.validate_non_negative_integer!(:patience)

    compiler = opts |> Keyword.get(:compiler, EXLA) |> Options.validate_compiler!()
    architecture = Keyword.get(opts, :architecture, @architecture)
    Model.validate_name!(architecture)
    dropout_rate = opts |> Keyword.get(:dropout_rate, 0.2) |> Options.validate_dropout_rate!()
    vector_length = Keyword.get(opts, :vector_length, @default_vector_length)

    decision_config =
      opts
      |> Keyword.get(:decision_policy, :argmax)
      |> DecisionPolicy.training_config!()

    minimum_input_tokens =
      opts
      |> Keyword.get(:minimum_input_tokens, 0)
      |> DecisionPolicy.validate_minimum_input_tokens!()

    validate_training_configuration!(
      epochs,
      batch_size,
      validation_split,
      architecture,
      vector_length,
      minimum_input_tokens
    )

    data_stream = Enum.to_list(data_stream)

    if data_stream == [] do
      raise ArgumentError, "training data must not be empty"
    end

    validate_targets!(data_stream, target_mode)
    validate_partition_sizes!(length(data_stream), batch_size, validation_split)

    validate_threshold_optimization_partitions!(
      decision_config,
      data_stream,
      batch_size,
      validation_split,
      target_mode,
      target_threshold
    )

    clear_recompilation_counters()
    vectorization_concurrency = min(4, System.schedulers_online())

    tokenizer = Tokenizer.train(Stream.map(data_stream, & &1.text))
    tokenizer_metadata = Tokenizer.metadata(tokenizer)
    vocab_size = tokenizer_metadata.vocab_size
    pad_token_id = tokenizer_metadata.pad_token_id

    {train_examples, test_examples} =
      split_and_balance_data(
        data_stream,
        batch_size,
        validation_split,
        balance?,
        target_mode,
        target_threshold,
        seed
      )

    {train_data, train_labels} =
      build_batches(
        train_examples,
        tokenizer,
        vector_length,
        batch_size,
        target_mode,
        vectorization_concurrency,
        pad_token_id
      )

    {test_data, test_labels} =
      build_batches(
        test_examples,
        tokenizer,
        vector_length,
        batch_size,
        target_mode,
        vectorization_concurrency,
        pad_token_id
      )

    model_opts =
      [dropout_rate: dropout_rate, pad_token_id: pad_token_id]
      |> put_seed(seed)

    build_model_fn = fn ->
      Model.build(architecture, vocab_size, model_opts)
    end

    optimizer = Polaris.Optimizers.adamw(learning_rate: learning_rate, decay: decay)

    train_model =
      Axon.Loop.trainer(
        build_model_fn.(),
        :categorical_cross_entropy,
        optimizer,
        put_seed([], seed)
      )

    best_checkpoint =
      checkpointed_loop_run(%{
        train_model: train_model,
        build_model_fn: build_model_fn,
        train_data: Stream.zip(train_data, train_labels),
        test_data: Stream.zip(test_data, test_labels),
        epochs: epochs,
        patience: patience,
        compiler: compiler,
        decision_config: decision_config,
        minimum_input_tokens: minimum_input_tokens,
        target_mode: target_mode,
        target_threshold: target_threshold,
        pad_token_id: pad_token_id
      })

    %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: best_checkpoint.checkpoint.step_state.model_state,
      accuracy: best_checkpoint.accuracy,
      epoch: best_checkpoint.epoch,
      vector_length: vector_length,
      vocab_size: vocab_size,
      pad_token_id: pad_token_id,
      unk_token_id: tokenizer_metadata.unk_token_id,
      labels: labels,
      architecture: architecture,
      model_revision: Model.revision(),
      learning_rate: learning_rate,
      dropout_rate: dropout_rate,
      decision_policy: best_checkpoint.decision_policy,
      minimum_input_tokens: minimum_input_tokens,
      target_mode: target_mode,
      target_threshold: target_threshold
    }
  end

  @doc false
  def split_and_balance_data(
        data,
        batch_size,
        validation_split,
        balance?,
        target_mode,
        target_threshold,
        seed
      ) do
    {train_examples_count, validation_examples_count} =
      partition_example_counts(length(data), batch_size, validation_split)

    {class_0, class_1} =
      Enum.split_with(data, &(target_class(&1.label, target_mode, target_threshold) == 0))

    class_0 = shuffle(class_0, seed, :partition_class_0)
    class_1 = shuffle(class_1, seed, :partition_class_1)

    stratified? =
      length(class_0) >= 2 and length(class_1) >= 2 and
        train_examples_count >= 2 and validation_examples_count >= 2

    minimum_per_class = if stratified?, do: 1, else: 0

    validation_class_1_count =
      partition_class_1_count(
        validation_examples_count,
        length(class_0) - minimum_per_class,
        length(class_1) - minimum_per_class,
        minimum_per_class,
        minimum_per_class
      )

    validation_class_0_count = validation_examples_count - validation_class_1_count

    {validation_class_0, remaining_class_0} = Enum.split(class_0, validation_class_0_count)
    {validation_class_1, remaining_class_1} = Enum.split(class_1, validation_class_1_count)

    train_class_1_count =
      partition_class_1_count(
        train_examples_count,
        length(remaining_class_0),
        length(remaining_class_1),
        minimum_per_class,
        minimum_per_class
      )

    train_class_0_count = train_examples_count - train_class_1_count

    train_data =
      Enum.take(remaining_class_0, train_class_0_count) ++
        Enum.take(remaining_class_1, train_class_1_count)

    validation_data =
      (validation_class_0 ++ validation_class_1)
      |> shuffle(seed, :validation_partition)

    train_data =
      if balance? do
        train_data
        |> balance_data(target_mode, target_threshold)
        |> shuffle(seed, :balanced_training)
      else
        shuffle(train_data, seed, :training_partition)
      end

    {train_data, validation_data}
  end

  defp build_batches(
         examples,
         tokenizer,
         vector_length,
         batch_size,
         target_mode,
         vectorization_concurrency,
         pad_token_id
       ) do
    examples
    |> Stream.chunk_every(batch_size, batch_size, :discard)
    |> Task.async_stream(
      fn batch -> build_batch(batch, tokenizer, vector_length, target_mode, pad_token_id) end,
      ordered: true,
      max_concurrency: vectorization_concurrency,
      timeout: :infinity
    )
    |> Enum.map(fn {:ok, batch} -> batch end)
    |> Enum.unzip()
  end

  defp build_batch(examples, tokenizer, vector_length, target_mode, pad_token_id) do
    data =
      examples
      |> Enum.map(fn example ->
        Vectorizer.build(tokenizer, example.text, vector_length, pad_token_id)
      end)
      |> Nx.tensor(type: :u16)

    labels =
      examples
      |> Enum.map(& &1.label)
      |> Nx.tensor()
      |> target_tensor(target_mode)

    {data, labels}
  end

  defp balance_data(data, target_mode, target_threshold) do
    {class_0, class_1} =
      Enum.split_with(data, &(target_class(&1.label, target_mode, target_threshold) == 0))

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
           decision_config: decision_config,
           minimum_input_tokens: minimum_input_tokens,
           target_mode: target_mode,
           target_threshold: target_threshold,
           pad_token_id: pad_token_id
         } = map,
         opts \\ []
       ) do
    init_state = Map.get(map, :init_state, Axon.ModelState.empty())

    opts =
      opts
      |> Keyword.put(:epochs, epochs)
      |> Keyword.put(:compiler, compiler)

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
              target_mode,
              target_threshold,
              pad_token_id,
              %{strategy: :argmax},
              0
            )

          %{
            epoch: epoch,
            accuracy: metrics.accuracy,
            validation_score: metrics.accuracy,
            checkpoint: checkpoint
          }
        end)
        |> Enum.max_by(& &1.validation_score)

      decision_policy =
        resolve_decision_policy(
          decision_config,
          predict_fn,
          best_checkpoint.checkpoint.step_state.model_state,
          test_data,
          target_mode,
          target_threshold,
          pad_token_id,
          minimum_input_tokens
        )

      deployment_metrics =
        calculate_validation_metrics(
          predict_fn,
          best_checkpoint.checkpoint.step_state.model_state,
          test_data,
          target_mode,
          target_threshold,
          pad_token_id,
          decision_policy,
          minimum_input_tokens
        )

      best_checkpoint
      |> Map.put(:accuracy, deployment_metrics.accuracy)
      |> Map.put(:decision_policy, decision_policy)
    end)
  end

  defp calculate_validation_metrics(
         predict_fn,
         trained_model_state,
         test_data,
         target_mode,
         target_threshold,
         pad_token_id,
         decision_policy,
         minimum_input_tokens
       ) do
    test_data
    |> prediction_batches(predict_fn, trained_model_state)
    |> calculate_prediction_metrics(
      target_mode,
      target_threshold,
      pad_token_id,
      decision_policy,
      minimum_input_tokens
    )
  end

  defp prediction_batches(test_data, predict_fn, trained_model_state) do
    Stream.map(test_data, fn {inputs, labels} ->
      {predict_fn.(trained_model_state, inputs), inputs, labels}
    end)
  end

  defp calculate_prediction_metrics(
         prediction_batches,
         target_mode,
         target_threshold,
         pad_token_id,
         decision_policy,
         minimum_input_tokens
       ) do
    {correct, total, false_positives, true_negatives, insufficient_inputs} =
      Enum.reduce(prediction_batches, {0, 0, 0, 0, 0}, fn {preds, inputs, labels},
                                                          {correct_acc, count_acc, fp_acc, tn_acc,
                                                           insufficient_acc} ->
        pred_labels = prediction_labels(preds, decision_policy)
        true_labels = target_labels(labels, target_mode, target_threshold)
        eligible_inputs = eligible_inputs(inputs, pad_token_id, minimum_input_tokens)

        {correct, eligible_count, false_positives, true_negatives} =
          validation_counts(pred_labels, true_labels, eligible_inputs)

        batch_size = Nx.shape(inputs) |> elem(0)

        {
          correct_acc + correct,
          count_acc + eligible_count,
          fp_acc + false_positives,
          tn_acc + true_negatives,
          insufficient_acc + batch_size - eligible_count
        }
      end)

    if total == 0 do
      raise ArgumentError,
            ":minimum_input_tokens excludes every validation example; lower it or provide " <>
              "longer validation inputs"
    end

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
      true_negatives: true_negatives,
      insufficient_inputs: insufficient_inputs
    }
  end

  defp validation_counts(pred_labels, true_labels, eligible_inputs) do
    correct =
      pred_labels
      |> Nx.equal(true_labels)
      |> Nx.logical_and(eligible_inputs)
      |> Nx.sum()

    eligible_count = Nx.sum(eligible_inputs)
    predicted_positive? = Nx.equal(pred_labels, 1)
    true_negative? = Nx.equal(true_labels, 0)

    false_positives =
      predicted_positive?
      |> Nx.logical_and(true_negative?)
      |> Nx.logical_and(eligible_inputs)
      |> Nx.sum()

    true_negatives =
      pred_labels
      |> Nx.equal(0)
      |> Nx.logical_and(true_negative?)
      |> Nx.logical_and(eligible_inputs)
      |> Nx.sum()

    [correct, eligible_count, false_positives, true_negatives]
    |> Nx.stack()
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_flat_list()
    |> List.to_tuple()
  end

  defp prediction_labels(preds, %{strategy: :argmax}), do: Nx.argmax(preds, axis: -1)

  defp prediction_labels(preds, %{
         strategy: :positive_threshold,
         positive_threshold: threshold
       }) do
    batch_size = Nx.shape(preds) |> elem(0)

    positive? =
      preds
      |> Nx.slice([0, 1], [batch_size, 1])
      |> Nx.squeeze(axes: [1])
      |> Nx.greater_equal(threshold)

    Nx.select(
      positive?,
      Nx.broadcast(1, Nx.shape(positive?)),
      Nx.broadcast(0, Nx.shape(positive?))
    )
  end

  defp eligible_inputs(inputs, pad_token_id, minimum_input_tokens) do
    inputs
    |> BinClass.Padding.active_token_counts(pad_token_id)
    |> Nx.greater_equal(minimum_input_tokens)
  end

  defp target_tensor(targets, :hard) do
    targets
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor([0, 1]))
  end

  defp target_tensor(targets, :soft) do
    positive_targets =
      targets
      |> Nx.as_type(:f32)
      |> Nx.new_axis(-1)

    negative_targets = Nx.subtract(1.0, positive_targets)
    Nx.concatenate([negative_targets, positive_targets], axis: -1)
  end

  defp target_labels(labels, :hard, _target_threshold) do
    Nx.argmax(labels, axis: -1)
  end

  defp target_labels(labels, :soft, target_threshold) do
    batch_size = Nx.shape(labels) |> elem(0)

    labels
    |> Nx.slice([0, 1], [batch_size, 1])
    |> Nx.squeeze(axes: [1])
    |> Nx.greater_equal(target_threshold)
  end

  defp target_class(label, :hard, _target_threshold) do
    if label == 0, do: 0, else: 1
  end

  defp target_class(label, :soft, target_threshold) do
    if label >= target_threshold, do: 1, else: 0
  end

  defp validate_target_options!(target_mode, target_threshold) do
    unless target_mode in [:hard, :soft] do
      raise ArgumentError,
            ":target_mode must be :hard or :soft, got: #{inspect(target_mode)}"
    end

    unless is_number(target_threshold) and target_threshold >= 0 and target_threshold <= 1 do
      raise ArgumentError,
            ":target_threshold must be a number between 0 and 1, got: #{inspect(target_threshold)}"
    end
  end

  defp validate_targets!(data, target_mode) do
    Enum.each(data, fn row ->
      label = Map.fetch!(row, :label)

      valid? =
        case target_mode do
          :hard -> is_number(label) and (label == 0 or label == 1)
          :soft -> is_number(label) and label >= 0 and label <= 1
        end

      unless valid? do
        expected =
          if target_mode == :soft,
            do: "a number between 0 and 1",
            else: "0 or 1"

        raise ArgumentError,
              "invalid #{target_mode} target #{inspect(label)}; expected #{expected}"
      end
    end)
  end

  defp validate_seed!(nil), do: :ok
  defp validate_seed!(seed) when is_integer(seed), do: :ok

  defp validate_seed!(seed) do
    raise ArgumentError, ":seed must be an integer or nil, got: #{inspect(seed)}"
  end

  defp validate_training_configuration!(
         epochs,
         batch_size,
         validation_split,
         architecture,
         vector_length,
         minimum_input_tokens
       ) do
    unless is_integer(epochs) and epochs > 0 do
      raise ArgumentError, ":epochs must be a positive integer, got: #{inspect(epochs)}"
    end

    unless is_integer(batch_size) and batch_size > 0 do
      raise ArgumentError, ":batch_size must be a positive integer, got: #{inspect(batch_size)}"
    end

    unless is_number(validation_split) and validation_split > 0 and validation_split < 1 do
      raise ArgumentError,
            ":validation_split must be a number greater than 0 and less than 1, " <>
              "got: #{inspect(validation_split)}"
    end

    Model.validate_vector_length!(architecture, vector_length)

    if minimum_input_tokens > vector_length do
      raise ArgumentError,
            ":minimum_input_tokens cannot exceed :vector_length, got " <>
              "#{minimum_input_tokens} and #{vector_length}"
    end
  end

  defp validate_partition_sizes!(data_size, batch_size, validation_split) do
    {train_examples_count, validation_examples_count} =
      partition_example_counts(data_size, batch_size, validation_split)

    train_batches_count = div(train_examples_count, batch_size)
    validation_batches_count = div(validation_examples_count, batch_size)

    if train_batches_count == 0 or validation_batches_count == 0 do
      raise ArgumentError,
            "training configuration must produce at least one full training batch and " <>
              "one full validation batch, but #{data_size} examples with batch_size " <>
              "#{batch_size} and validation_split #{inspect(validation_split)} produce " <>
              "#{train_batches_count} training batches and #{validation_batches_count} " <>
              "validation batches; reduce :batch_size, adjust :validation_split, or " <>
              "provide more data"
    end
  end

  defp validate_threshold_optimization_partitions!(
         {:fixed, _policy},
         _data,
         _batch_size,
         _validation_split,
         _target_mode,
         _target_threshold
       ),
       do: :ok

  defp validate_threshold_optimization_partitions!(
         {:optimize, _optimization},
         data,
         batch_size,
         validation_split,
         target_mode,
         target_threshold
       ) do
    {class_0_count, class_1_count} = class_counts(data, target_mode, target_threshold)

    if class_0_count < 2 or class_1_count < 2 do
      raise ArgumentError,
            "threshold optimization needs at least two examples from each class so training " <>
              "and validation can each contain both classes; got #{class_0_count} negative " <>
              "and #{class_1_count} positive examples"
    end

    {train_examples_count, validation_examples_count} =
      partition_example_counts(length(data), batch_size, validation_split)

    if train_examples_count < 2 or validation_examples_count < 2 do
      raise ArgumentError,
            "threshold optimization needs training and validation partitions that can each " <>
              "contain both classes; batch_size #{batch_size} and validation_split " <>
              "#{inspect(validation_split)} produce #{train_examples_count} training and " <>
              "#{validation_examples_count} validation examples"
    end

    :ok
  end

  defp partition_example_counts(data_size, batch_size, validation_split) do
    batch_count = div(data_size, batch_size)
    train_batches_count = floor(batch_count * (1.0 - validation_split))
    validation_batches_count = batch_count - train_batches_count

    {train_batches_count * batch_size, validation_batches_count * batch_size}
  end

  defp class_counts(data, target_mode, target_threshold) do
    Enum.reduce(data, {0, 0}, fn row, {class_0_count, class_1_count} ->
      case target_class(row.label, target_mode, target_threshold) do
        0 -> {class_0_count + 1, class_1_count}
        1 -> {class_0_count, class_1_count + 1}
      end
    end)
  end

  defp partition_class_1_count(partition_size, class_0_count, class_1_count, min_0, min_1) do
    minimum = max(min_1, partition_size - class_0_count)
    maximum = min(class_1_count, partition_size - min_0)

    proportional =
      round(partition_size * class_1_count / (class_0_count + class_1_count))

    proportional
    |> max(minimum)
    |> min(maximum)
  end

  defp shuffle(data, nil, _salt), do: Enum.shuffle(data)

  defp shuffle(data, seed, salt) do
    data
    |> Enum.with_index()
    |> Enum.sort_by(fn {row, index} ->
      identity = Map.get(row, :id, {Map.get(row, :text), Map.get(row, :label)})
      :crypto.hash(:sha256, :erlang.term_to_binary({seed, salt, identity, index}))
    end)
    |> Enum.map(&elem(&1, 0))
  end

  defp put_seed(opts, nil), do: opts
  defp put_seed(opts, seed), do: Keyword.put(opts, :seed, seed)

  defp resolve_decision_policy(
         {:fixed, policy},
         _predict_fn,
         _trained_model_state,
         _test_data,
         _target_mode,
         _target_threshold,
         _pad_token_id,
         _minimum_input_tokens
       ),
       do: policy

  defp resolve_decision_policy(
         {:optimize, optimization},
         predict_fn,
         trained_model_state,
         test_data,
         target_mode,
         target_threshold,
         pad_token_id,
         minimum_input_tokens
       ) do
    prediction_batches =
      test_data
      |> prediction_batches(predict_fn, trained_model_state)
      |> Enum.to_list()

    calibration_opts =
      optimization
      |> Map.to_list()
      |> Keyword.put(:eligible, :eligible)

    result =
      prediction_batches
      |> calibration_rows(
        target_mode,
        target_threshold,
        pad_token_id,
        minimum_input_tokens
      )
      |> BinClass.Evaluation.calibrate(calibration_opts)

    %{
      strategy: :positive_threshold,
      positive_threshold: result.threshold,
      optimization:
        Map.merge(optimization, %{
          validation_score: result.score,
          validation_accuracy: result.accuracy,
          validation_balanced_accuracy: result.balanced_accuracy,
          validation_f1: result.f1,
          validation_false_positive_rate: result.false_positive_rate
        })
    }
  end

  defp calibration_rows(
         prediction_batches,
         target_mode,
         target_threshold,
         pad_token_id,
         minimum_input_tokens
       ) do
    Enum.flat_map(prediction_batches, fn {predictions, inputs, labels} ->
      batch_size = Nx.shape(inputs) |> elem(0)

      positive_probabilities =
        predictions
        |> Nx.slice([0, 1], [batch_size, 1])
        |> Nx.squeeze(axes: [1])

      eligible =
        inputs
        |> BinClass.Padding.active_token_counts(pad_token_id)
        |> Nx.greater_equal(minimum_input_tokens)

      true_labels = target_labels(labels, target_mode, target_threshold)

      [positive_probabilities, Nx.as_type(eligible, :f32), Nx.as_type(true_labels, :f32)]
      |> Nx.stack(axis: 1)
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_list()
      |> Enum.map(fn [probability, eligible?, positive?] ->
        %{
          probability: probability,
          eligible: eligible? == 1.0,
          label: positive? == 1.0
        }
      end)
    end)
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
