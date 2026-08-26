defmodule BinClass.Training.Dataset do
  @moduledoc false

  alias BinClass.Training.Config
  alias BinClass.Vectorizer

  @doc false
  def prepare!(data_stream, %Config{} = config) do
    data = Enum.to_list(data_stream)

    if data == [] do
      raise ArgumentError, "training data must not be empty"
    end

    validate_targets!(data, config.target_mode)
    validate_partition_sizes!(length(data), config.batch_size, config.validation_split)
    validate_threshold_optimization_partitions!(data, config)

    data
  end

  @doc false
  def split_and_balance(data, %Config{} = config) do
    {training_examples_count, validation_examples_count} =
      partition_example_counts(length(data), config.batch_size, config.validation_split)

    {class_0, class_1} =
      Enum.split_with(data, &(target_class(&1.label, config) == 0))

    class_0 = shuffle(class_0, config.seed, :partition_class_0)
    class_1 = shuffle(class_1, config.seed, :partition_class_1)

    stratified? =
      length(class_0) >= 2 and length(class_1) >= 2 and
        training_examples_count >= 2 and validation_examples_count >= 2

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

    training_class_1_count =
      partition_class_1_count(
        training_examples_count,
        length(remaining_class_0),
        length(remaining_class_1),
        minimum_per_class,
        minimum_per_class
      )

    training_class_0_count = training_examples_count - training_class_1_count

    training_data =
      Enum.take(remaining_class_0, training_class_0_count) ++
        Enum.take(remaining_class_1, training_class_1_count)

    validation_data =
      (validation_class_0 ++ validation_class_1)
      |> shuffle(config.seed, :validation_partition)

    training_data =
      if config.balance do
        training_data
        |> balance_data(config)
        |> shuffle(config.seed, :balanced_training)
      else
        shuffle(training_data, config.seed, :training_partition)
      end

    {training_data, validation_data}
  end

  @doc false
  def build_batches(examples, tokenizer, %Config{} = config, pad_token_id) do
    vectorization_concurrency = min(4, System.schedulers_online())

    examples
    |> Stream.chunk_every(config.batch_size, config.batch_size, :discard)
    |> Task.async_stream(
      fn batch -> build_batch(batch, tokenizer, config, pad_token_id) end,
      ordered: true,
      max_concurrency: vectorization_concurrency,
      timeout: :infinity
    )
    |> Enum.map(fn {:ok, batch} -> batch end)
    |> Enum.unzip()
  end

  @doc false
  def target_labels(labels, %Config{target_mode: :hard}) do
    Nx.argmax(labels, axis: -1)
  end

  def target_labels(labels, %Config{target_mode: :soft, target_threshold: target_threshold}) do
    batch_size = Nx.shape(labels) |> elem(0)

    labels
    |> Nx.slice([0, 1], [batch_size, 1])
    |> Nx.squeeze(axes: [1])
    |> Nx.greater_equal(target_threshold)
  end

  defp build_batch(examples, tokenizer, config, pad_token_id) do
    data =
      examples
      |> Enum.map(fn example ->
        Vectorizer.build(tokenizer, example.text, config.vector_length, pad_token_id)
      end)
      |> Nx.tensor(type: :u16)

    labels =
      examples
      |> Enum.map(& &1.label)
      |> Nx.tensor()
      |> target_tensor(config.target_mode)

    {data, labels}
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

  defp balance_data(data, config) do
    {class_0, class_1} = Enum.split_with(data, &(target_class(&1.label, config) == 0))
    class_0_count = length(class_0)
    class_1_count = length(class_1)

    cond do
      class_0_count > class_1_count and class_1_count > 0 ->
        oversampled_class_1 = Stream.cycle(class_1) |> Enum.take(class_0_count)
        class_0 ++ oversampled_class_1

      class_1_count > class_0_count and class_0_count > 0 ->
        oversampled_class_0 = Stream.cycle(class_0) |> Enum.take(class_1_count)
        class_1 ++ oversampled_class_0

      true ->
        data
    end
  end

  defp target_class(label, %Config{target_mode: :hard}) do
    if label == 0, do: 0, else: 1
  end

  defp target_class(label, %Config{
         target_mode: :soft,
         target_threshold: target_threshold
       }) do
    if label >= target_threshold, do: 1, else: 0
  end

  defp validate_targets!(data, target_mode) do
    Enum.each(data, &validate_target!(&1, target_mode))
  end

  defp validate_target!(row, target_mode) do
    label = Map.fetch!(row, :label)

    unless valid_target?(label, target_mode) do
      raise ArgumentError,
            "invalid #{target_mode} target #{inspect(label)}; " <>
              "expected #{target_expectation(target_mode)}"
    end
  end

  defp valid_target?(label, :hard), do: is_number(label) and (label == 0 or label == 1)

  defp valid_target?(label, :soft),
    do: is_number(label) and label >= 0 and label <= 1

  defp target_expectation(:hard), do: "0 or 1"
  defp target_expectation(:soft), do: "a number between 0 and 1"

  defp validate_partition_sizes!(data_size, batch_size, validation_split) do
    {training_examples_count, validation_examples_count} =
      partition_example_counts(data_size, batch_size, validation_split)

    training_batches_count = div(training_examples_count, batch_size)
    validation_batches_count = div(validation_examples_count, batch_size)

    if training_batches_count == 0 or validation_batches_count == 0 do
      raise ArgumentError,
            "training configuration must produce at least one full training batch and " <>
              "one full validation batch, but #{data_size} examples with batch_size " <>
              "#{batch_size} and validation_split #{inspect(validation_split)} produce " <>
              "#{training_batches_count} training batches and #{validation_batches_count} " <>
              "validation batches; reduce :batch_size, adjust :validation_split, or " <>
              "provide more data"
    end
  end

  defp validate_threshold_optimization_partitions!(
         _data,
         %Config{decision_config: {:fixed, _policy}}
       ),
       do: :ok

  defp validate_threshold_optimization_partitions!(
         data,
         %Config{decision_config: {:optimize, _optimization}} = config
       ) do
    {class_0_count, class_1_count} = class_counts(data, config)

    if class_0_count < 2 or class_1_count < 2 do
      raise ArgumentError,
            "threshold optimization needs at least two examples from each class so training " <>
              "and validation can each contain both classes; got #{class_0_count} negative " <>
              "and #{class_1_count} positive examples"
    end

    {training_examples_count, validation_examples_count} =
      partition_example_counts(length(data), config.batch_size, config.validation_split)

    if training_examples_count < 2 or validation_examples_count < 2 do
      raise ArgumentError,
            "threshold optimization needs training and validation partitions that can each " <>
              "contain both classes; batch_size #{config.batch_size} and validation_split " <>
              "#{inspect(config.validation_split)} produce #{training_examples_count} training " <>
              "and #{validation_examples_count} validation examples"
    end

    :ok
  end

  defp partition_example_counts(data_size, batch_size, validation_split) do
    batch_count = div(data_size, batch_size)
    training_batches_count = floor(batch_count * (1.0 - validation_split))
    validation_batches_count = batch_count - training_batches_count

    {training_batches_count * batch_size, validation_batches_count * batch_size}
  end

  defp class_counts(data, config) do
    Enum.reduce(data, {0, 0}, fn row, {class_0_count, class_1_count} ->
      case target_class(row.label, config) do
        0 -> {class_0_count + 1, class_1_count}
        1 -> {class_0_count, class_1_count + 1}
      end
    end)
  end

  defp partition_class_1_count(partition_size, class_0_count, class_1_count, min_0, min_1) do
    minimum = max(min_1, partition_size - class_0_count)
    maximum = min(class_1_count, partition_size - min_0)

    proportional = round(partition_size * class_1_count / (class_0_count + class_1_count))

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
end
