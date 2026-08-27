defmodule BinClass.Dataset do
  @moduledoc """
  Utilities for dataset inspection, summary statistics, and stratified splitting.
  """

  @doc """
  Computes summary statistics on a binary classification dataset.

  Calculates total count, class frequencies and distribution percentages,
  class balance ratio, character length metrics, and estimated word counts.

  ## Options

    * `:labels` - The pair of class names, e.g. `["0", "1"]` or `%{0 => "neg", 1 => "pos"}`.
      Defaults to `["0", "1"]`.
    * `:target_threshold` - Float boundary for soft labels. Defaults to `0.5`.

  ## Examples

      data = [
        %{text: "excellent movie", label: 1},
        %{text: "awful experience", label: 0}
      ]

      summary = BinClass.Dataset.summary(data)
  """
  def summary(data, opts \\ []) do
    opts = Keyword.validate!(opts, labels: ["0", "1"], target_threshold: 0.5)
    {label_0, label_1} = extract_labels(Keyword.fetch!(opts, :labels))
    threshold = Keyword.fetch!(opts, :target_threshold)

    data_list = Enum.to_list(data)

    if data_list == [] do
      raise ArgumentError, "dataset must not be empty"
    end

    {class_0_count, class_1_count, char_lens, w_counts} =
      accumulate_summary_metrics(data_list, label_0, label_1, threshold)

    total = length(data_list)
    balance_ratio = compute_balance_ratio(class_0_count, class_1_count)

    %{
      total: total,
      class_counts: %{label_0 => class_0_count, label_1 => class_1_count},
      class_distribution: %{
        label_0 => safe_divide(class_0_count, total),
        label_1 => safe_divide(class_1_count, total)
      },
      balance_ratio: balance_ratio,
      character_length: compute_stats(char_lens),
      word_count: compute_stats(w_counts),
      labels: [label_0, label_1]
    }
  end

  @doc """
  Formats a dataset summary map into a human-readable ASCII report.
  """
  def format_summary(summary) when is_map(summary) do
    [label_0, label_1] = summary.labels
    c0 = Map.get(summary.class_counts, label_0, 0)
    c1 = Map.get(summary.class_counts, label_1, 0)
    p0 = Map.get(summary.class_distribution, label_0, 0.0) * 100.0
    p1 = Map.get(summary.class_distribution, label_1, 0.0) * 100.0
    chars = summary.character_length
    words = summary.word_count

    """
    ============================= Dataset Summary =============================
    Total Examples:       #{summary.total}
    Class Breakdown:
      * #{label_0}: #{c0} (#{Float.round(p0, 2)}%)
      * #{label_1}: #{c1} (#{Float.round(p1, 2)}%)
    Balance Ratio:        #{Float.round(summary.balance_ratio, 4)}

    Text Length (characters):
      Min: #{chars.min}  |  Max: #{chars.max}  |  Mean: #{chars.mean}  |  Median: #{chars.median}

    Word Count (estimated):
      Min: #{words.min}  |  Max: #{words.max}  |  Mean: #{words.mean}  |  Median: #{words.median}
    ===========================================================================
    """
    |> String.trim_trailing()
  end

  @doc """
  Splits a dataset into training and validation (or training, validation, and test) sets.

  Supports stratified splitting (`stratify: true`, the default) to preserve class
  proportions across all partitions, as well as deterministic random splitting with `:seed`.

  ## Options

    * `:validation_split` - Fraction for validation partition (`0.0 < ratio < 1.0`). Defaults to `0.1`.
    * `:test_split` - Optional fraction for test partition. When provided, returns `{train, val, test}`.
    * `:stratify` - Whether to maintain class balance across splits. Defaults to `true`.
    * `:seed` - Optional integer for reproducible random shuffle ordering.
    * `:labels` - Expected class labels for stratification. Defaults to `["0", "1"]`.
    * `:target_threshold` - Boundary for soft target binarization. Defaults to `0.5`.

  ## Examples

      {train, val} = BinClass.Dataset.split(data, validation_split: 0.2, seed: 42)

      {train, val, test} = BinClass.Dataset.split(data, validation_split: 0.1, test_split: 0.1, seed: 42)
  """
  def split(data, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        validation_split: 0.1,
        test_split: nil,
        stratify: true,
        labels: ["0", "1"],
        target_threshold: 0.5,
        seed: nil
      )

    val_ratio = Keyword.fetch!(opts, :validation_split)
    test_ratio = Keyword.get(opts, :test_split)
    stratify? = Keyword.fetch!(opts, :stratify)
    seed = Keyword.get(opts, :seed)
    {label_0, label_1} = extract_labels(Keyword.fetch!(opts, :labels))
    threshold = Keyword.fetch!(opts, :target_threshold)

    validate_ratios!(val_ratio, test_ratio)
    data_list = Enum.to_list(data)

    if data_list == [] do
      raise ArgumentError, "cannot split an empty dataset"
    end

    if stratify? do
      stratified_split(data_list, val_ratio, test_ratio, label_0, label_1, threshold, seed)
    else
      random_split(data_list, val_ratio, test_ratio, seed)
    end
  end

  defp accumulate_summary_metrics(data_list, label_0, label_1, threshold) do
    Enum.reduce(data_list, {0, 0, [], []}, fn item, {c0, c1, char_lens, w_counts} ->
      {text, raw_label} = extract_text_and_label(item)
      bin_label = normalize_label(raw_label, label_0, label_1, threshold)
      char_len = String.length(text)
      words = text |> String.split(~r/\s+/, trim: true) |> length()

      {
        if(bin_label == 0, do: c0 + 1, else: c0),
        if(bin_label == 1, do: c1 + 1, else: c1),
        [char_len | char_lens],
        [words | w_counts]
      }
    end)
  end

  defp compute_balance_ratio(class_0_count, class_1_count) do
    min_count = min(class_0_count, class_1_count)
    max_count = max(class_0_count, class_1_count)
    safe_divide(min_count, max_count)
  end

  defp compute_stats([]) do
    %{min: 0, max: 0, mean: 0.0, median: 0.0}
  end

  defp compute_stats(numbers) do
    sorted = Enum.sort(numbers)
    count = length(sorted)
    min_val = List.first(sorted)
    max_val = List.last(sorted)
    sum = Enum.sum(sorted)
    mean = sum / count
    median = calculate_median(sorted, count)

    %{
      min: min_val,
      max: max_val,
      mean: Float.round(mean, 2),
      median: Float.round(median, 2)
    }
  end

  defp calculate_median(sorted, count) do
    if rem(count, 2) == 1 do
      Enum.at(sorted, div(count, 2)) * 1.0
    else
      mid = div(count, 2)
      (Enum.at(sorted, mid - 1) + Enum.at(sorted, mid)) / 2.0
    end
  end

  defp stratified_split(data, val_ratio, test_ratio, label_0, label_1, threshold, seed) do
    {class_0, class_1} =
      Enum.split_with(data, fn item ->
        {_, label} = extract_text_and_label(item)
        normalize_label(label, label_0, label_1, threshold) == 0
      end)

    c0_shuffled = maybe_shuffle(class_0, seed, 0)
    c1_shuffled = maybe_shuffle(class_1, seed, 1)

    case test_ratio do
      nil ->
        {train_0, val_0} = split_partition(c0_shuffled, val_ratio)
        {train_1, val_1} = split_partition(c1_shuffled, val_ratio)
        {maybe_shuffle(train_0 ++ train_1, seed, 2), maybe_shuffle(val_0 ++ val_1, seed, 3)}

      test_r when is_number(test_r) ->
        {train_0, val_0, test_0} = split_3_partitions(c0_shuffled, val_ratio, test_r)
        {train_1, val_1, test_1} = split_3_partitions(c1_shuffled, val_ratio, test_r)

        {
          maybe_shuffle(train_0 ++ train_1, seed, 2),
          maybe_shuffle(val_0 ++ val_1, seed, 3),
          maybe_shuffle(test_0 ++ test_1, seed, 4)
        }
    end
  end

  defp random_split(data, val_ratio, test_ratio, seed) do
    shuffled = maybe_shuffle(data, seed, 0)

    case test_ratio do
      nil -> split_partition(shuffled, val_ratio)
      test_r when is_number(test_r) -> split_3_partitions(shuffled, val_ratio, test_r)
    end
  end

  defp split_partition(items, ratio) do
    total = length(items)
    part_size = round(total * ratio)
    val = Enum.take(items, part_size)
    train = Enum.drop(items, part_size)
    {train, val}
  end

  defp split_3_partitions(items, val_ratio, test_ratio) do
    total = length(items)
    val_size = round(total * val_ratio)
    test_size = round(total * test_ratio)

    val = Enum.slice(items, 0, val_size)
    test = Enum.slice(items, val_size, test_size)
    train = Enum.drop(items, val_size + test_size)

    {train, val, test}
  end

  defp maybe_shuffle(items, nil, _offset), do: items

  defp maybe_shuffle(items, seed, offset) when is_integer(seed) do
    :rand.seed(:exsss, {seed + offset, seed + offset, seed + offset})
    Enum.shuffle(items)
  end

  defp validate_ratios!(val_ratio, nil) do
    if !(is_number(val_ratio) and val_ratio > 0.0 and val_ratio < 1.0) do
      raise ArgumentError,
            ":validation_split must be between 0.0 and 1.0, got: #{inspect(val_ratio)}"
    end
  end

  defp validate_ratios!(val_ratio, test_ratio) when is_number(test_ratio) do
    validate_ratios!(val_ratio, nil)

    if !(test_ratio > 0.0 and test_ratio < 1.0 and val_ratio + test_ratio < 1.0) do
      raise ArgumentError,
            ":test_split must be positive and :validation_split + :test_split must be < 1.0, got #{val_ratio} + #{test_ratio}"
    end
  end

  defp extract_text_and_label(%{text: text, label: label}), do: {text, label}
  defp extract_text_and_label(%{"text" => text, "label" => label}), do: {text, label}
  defp extract_text_and_label({text, label}) when is_binary(text), do: {text, label}

  defp extract_text_and_label(other) do
    raise ArgumentError,
          "Invalid dataset item #{inspect(other)}. Expected map with :text and :label or {text, label} tuple."
  end

  defp normalize_label(label, _label_0, _label_1, threshold) when is_float(label) do
    if label >= threshold, do: 1, else: 0
  end

  defp normalize_label(label, _label_0, _label_1, _threshold) when is_integer(label) do
    if label == 1, do: 1, else: 0
  end

  defp normalize_label(label, label_0, label_1, _threshold) when is_binary(label) do
    cond do
      label == label_1 -> 1
      label == label_0 -> 0
      label in ["1", "true", "positive"] -> 1
      true -> 0
    end
  end

  defp normalize_label(label, label_0, label_1, threshold) when is_atom(label) do
    normalize_label(Atom.to_string(label), label_0, label_1, threshold)
  end

  defp extract_labels([label_0, label_1]), do: {label_0, label_1}
  defp extract_labels(%{0 => label_0, 1 => label_1}), do: {label_0, label_1}
  defp extract_labels(%{"0" => label_0, "1" => label_1}), do: {label_0, label_1}

  defp safe_divide(_num, den) when den == 0 or den == 0.0, do: 0.0
  defp safe_divide(num, den), do: num / den
end
