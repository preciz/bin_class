defmodule BinClass.Evaluation do
  @moduledoc """
  Evaluates classifiers, positive-class probabilities, and calibrates decision thresholds.

  Supports evaluating trained `%BinClass.Classifier{}` structs directly on raw
  text datasets, formatting evaluation reports, and performing fixed-threshold
  evaluation and calibration over probability rows.
  """

  alias BinClass.Classifier

  @objectives [:accuracy, :balanced_accuracy, :f1]

  @common_defaults [
    probability: :probability,
    label: :label,
    target: nil,
    eligible: nil,
    objective: :accuracy,
    false_positive_cost: 0.0
  ]

  @selection_defaults Keyword.delete(@common_defaults, :probability)

  @doc """
  Evaluates a classifier, predictor, or probability rows.

  When passed a `%BinClass.Classifier{}` or predictor function and a dataset,
  evaluates predictions on the dataset and returns comprehensive classification metrics.

  When passed probability rows and options, evaluates the rows at a fixed threshold.
  """
  def evaluate(classifier_or_predictor_or_rows, data_or_opts \\ [], opts \\ [])

  def evaluate(%Classifier{} = classifier, data, opts) do
    evaluate_dataset(classifier, data, opts)
  end

  def evaluate(predictor_fn, data, opts) when is_function(predictor_fn, 1) and is_list(data) do
    evaluate_dataset(predictor_fn, data, opts)
  end

  def evaluate(rows, opts, []) do
    opts =
      opts
      |> Keyword.validate!([threshold: 0.5] ++ @common_defaults)
      |> validate_options!()

    threshold = validate_threshold!(Keyword.fetch!(opts, :threshold))
    normalized = normalize_rows(rows, opts)
    eligible = eligible_rows!(normalized)
    context = metric_context(normalized, eligible, opts)
    counts = count_predictions(eligible, threshold)

    metrics(threshold, counts, context)
  end

  @doc """
  Evaluates a classifier against a dataset of text and labels.
  """
  def evaluate_dataset(classifier_or_predictor, data, opts \\ [])

  def evaluate_dataset(%Classifier{} = classifier, data, opts) do
    opts = Keyword.validate!(opts, batch_size: 32, compiler: EXLA, target_threshold: nil)
    batch_size = Keyword.fetch!(opts, :batch_size)
    compiler = Keyword.fetch!(opts, :compiler)

    threshold =
      Keyword.get(opts, :target_threshold) || classifier.target_threshold || 0.5

    predictor = BinClass.compile_predictor(classifier, batch_size: batch_size, compiler: compiler)
    {label_0, label_1} = extract_labels(classifier.labels)

    normalized_data = normalize_dataset(data, label_0, label_1, threshold)

    if normalized_data == [] do
      raise ArgumentError, "evaluation data must not be empty"
    end

    texts = Enum.map(normalized_data, & &1.text)
    predictions = predictor.(texts)

    calculate_metrics(normalized_data, predictions, label_0, label_1)
  end

  def evaluate_dataset(predictor_fn, data, opts) when is_function(predictor_fn, 1) do
    opts = Keyword.validate!(opts, labels: ["0", "1"], target_threshold: 0.5)
    {label_0, label_1} = extract_labels(Keyword.fetch!(opts, :labels))
    threshold = Keyword.fetch!(opts, :target_threshold)

    normalized_data = normalize_dataset(data, label_0, label_1, threshold)

    if normalized_data == [] do
      raise ArgumentError, "evaluation data must not be empty"
    end

    texts = Enum.map(normalized_data, & &1.text)
    predictions = predictor_fn.(texts)
    predictions = if is_list(predictions), do: predictions, else: [predictions]

    calculate_metrics(normalized_data, predictions, label_0, label_1)
  end

  @doc """
  Formats an evaluation result into a human-readable text report.
  """
  def format_report(result) when is_map(result) do
    [label_0, label_1] = result.labels
    cm = result.confusion_matrix

    """
    ============================= Evaluation Report =============================
    Total Samples:        #{result.total}
    Accuracy:             #{format_percentage(result.accuracy)}
    Balanced Accuracy:    #{format_percentage(result.balanced_accuracy)}
    Precision (#{label_1}): #{format_percentage(result.precision)}
    Recall (#{label_1}):    #{format_percentage(result.recall)}
    Specificity (#{label_0}): #{format_percentage(result.specificity)}
    F1 Score:             #{format_percentage(result.f1_score)}
    Average Loss:         #{Float.round(result.loss, 4)}

    Confusion Matrix:
                          Predicted #{label_0}    Predicted #{label_1}
      Actual #{label_0}#{String.pad_leading("#{cm.tn} (TN)", 18)}#{String.pad_leading("#{cm.fp} (FP)", 24)}
      Actual #{label_1}#{String.pad_leading("#{cm.fn} (FN)", 18)}#{String.pad_leading("#{cm.tp} (TP)", 24)}
    =============================================================================
    """
    |> String.trim_trailing()
  end

  defp extract_labels([label_0, label_1]), do: {label_0, label_1}
  defp extract_labels(%{0 => label_0, 1 => label_1}), do: {label_0, label_1}
  defp extract_labels(%{"0" => label_0, "1" => label_1}), do: {label_0, label_1}

  defp normalize_dataset(data, label_0, label_1, threshold) do
    data
    |> Enum.map(fn item ->
      {text, raw_label} = extract_text_and_label(item)
      expected_class = normalize_eval_label(raw_label, label_0, label_1, threshold)
      target_prob = extract_target_prob(raw_label, threshold)

      %{
        text: text,
        expected_class: expected_class,
        target_prob: target_prob
      }
    end)
  end

  defp extract_text_and_label(%{text: text, label: label}), do: {text, label}
  defp extract_text_and_label(%{"text" => text, "label" => label}), do: {text, label}
  defp extract_text_and_label({text, label}) when is_binary(text), do: {text, label}

  defp extract_text_and_label(other) do
    raise ArgumentError,
          "Invalid evaluation item #{inspect(other)}. Expected map with :text and :label or {text, label} tuple."
  end

  defp normalize_eval_label(label, _label_0, _label_1, threshold) when is_float(label) do
    if label >= threshold, do: 1, else: 0
  end

  defp normalize_eval_label(label, _label_0, _label_1, _threshold) when is_integer(label) do
    if label == 1, do: 1, else: 0
  end

  defp normalize_eval_label(label, label_0, label_1, _threshold) when is_binary(label) do
    cond do
      label == label_1 -> 1
      label == label_0 -> 0
      label in ["1", "true", "positive"] -> 1
      true -> 0
    end
  end

  defp normalize_eval_label(label, label_0, label_1, threshold) when is_atom(label) do
    normalize_eval_label(Atom.to_string(label), label_0, label_1, threshold)
  end

  defp extract_target_prob(label, _threshold)
       when is_float(label) and label >= 0.0 and label <= 1.0,
       do: label

  defp extract_target_prob(label, _threshold) when is_integer(label),
    do: if(label == 1, do: 1.0, else: 0.0)

  defp extract_target_prob(label, threshold) do
    if normalize_eval_label(label, "0", "1", threshold) == 1, do: 1.0, else: 0.0
  end

  defp calculate_metrics(normalized_data, predictions, label_0, label_1) do
    pairs = Enum.zip(normalized_data, predictions)
    total = length(pairs)

    {counts, total_loss} =
      Enum.reduce(pairs, {%{tp: 0, fp: 0, tn: 0, fn: 0}, 0.0}, fn {item, pred}, {cm, loss_sum} ->
        {category, sample_loss} = evaluate_sample(item, pred, label_0, label_1)
        {Map.update!(cm, category, &(&1 + 1)), loss_sum + sample_loss}
      end)

    compute_metric_summary(counts, total_loss, total, label_0, label_1)
  end

  defp evaluate_sample(item, pred, label_0, label_1) do
    is_true_pos = item.expected_class == 1
    is_pred_pos = pred.label == label_1
    prob_pos = extract_probability_positive(pred, label_0, label_1, is_pred_pos)

    clipped_p = max(1.0e-7, min(1.0 - 1.0e-7, prob_pos))
    y = item.target_prob
    sample_loss = -(y * :math.log(clipped_p) + (1.0 - y) * :math.log(1.0 - clipped_p))
    category = classify_confusion_category(is_true_pos, is_pred_pos)

    {category, sample_loss}
  end

  defp extract_probability_positive(%{probabilities: %{} = probs}, label_0, label_1, _is_pred_pos) do
    case probs do
      %{^label_1 => p} -> p
      %{^label_0 => p0} -> 1.0 - p0
      _ -> 0.5
    end
  end

  defp extract_probability_positive(_pred, _label_0, _label_1, is_pred_pos) do
    if is_pred_pos, do: 0.99, else: 0.01
  end

  defp classify_confusion_category(true, true), do: :tp
  defp classify_confusion_category(false, true), do: :fp
  defp classify_confusion_category(false, false), do: :tn
  defp classify_confusion_category(true, false), do: :fn

  defp compute_metric_summary(cm, total_loss, total, label_0, label_1) do
    %{tp: tp, fp: fp, tn: tn, fn: fn_count} = cm

    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn_count)
    specificity = safe_divide(tn, tn + fp)
    balanced_accuracy = (recall + specificity) / 2.0
    f1_score = safe_divide(2.0 * precision * recall, precision + recall)
    avg_loss = safe_divide(total_loss, total)

    %{
      total: total,
      accuracy: accuracy,
      balanced_accuracy: balanced_accuracy,
      precision: precision,
      recall: recall,
      specificity: specificity,
      f1_score: f1_score,
      loss: avg_loss,
      confusion_matrix: cm,
      class_distribution: %{
        label_0 => tn + fp,
        label_1 => tp + fn_count
      },
      labels: [label_0, label_1]
    }
  end

  defp safe_divide(num, den) do
    if den == 0 or den == 0.0 do
      0.0
    else
      num / den
    end
  end

  defp format_percentage(val) when is_float(val) do
    "#{Float.round(val * 100.0, 2)}%"
  end

  # ============================================================================
  # THRESHOLD CALIBRATION & ROW EVALUATION
  # ============================================================================

  @doc """
  Selects the best positive-class threshold for the supplied rows.

  Accepts the same row accessors and objective options as `evaluate/2`, except
  for `:threshold`. Calibration requires at least one eligible example from
  each class.
  """
  def calibrate(rows, opts \\ []) do
    opts = opts |> Keyword.validate!(@common_defaults) |> validate_options!()
    normalized = normalize_rows(rows, opts)
    eligible = eligible_rows!(normalized)
    context = metric_context(normalized, eligible, opts)

    if context.negative_count == 0 or context.positive_count == 0 do
      raise ArgumentError, "calibration requires eligible examples from both classes"
    end

    initial_counts = %{
      true_positives: 0,
      true_negatives: context.negative_count,
      false_positives: 0,
      false_negatives: context.positive_count
    }

    groups =
      eligible
      |> Enum.group_by(& &1.probability)
      |> Enum.sort_by(&elem(&1, 0), :desc)

    boundaries = Enum.zip(groups, Enum.drop(groups, 1) ++ [nil])

    {results, _counts} =
      Enum.map_reduce(boundaries, initial_counts, fn {{probability, group}, next_group}, counts ->
        counts = Enum.reduce(group, counts, &mark_positive/2)

        lower_probability =
          case next_group do
            {next_probability, _group} -> next_probability
            nil -> 0.0
          end

        {metrics(midpoint(probability, lower_probability), counts, context), counts}
      end)

    results =
      case groups do
        [{1.0, _group} | _groups] ->
          results

        [{highest_probability, _group} | _groups] ->
          [metrics(midpoint(1.0, highest_probability), initial_counts, context) | results]
      end

    Enum.max_by(results, &metric_order/1)
  end

  @doc """
  Calibrates and compares named probability strategies.

  The required `:strategies` option is a non-empty list of
  `{strategy_name, probability_accessor}` pairs. All other options are passed
  to `calibrate/2`. Equivalent strategies are resolved in favor of the first
  one supplied.
  """
  def select(rows, opts) do
    opts =
      opts
      |> Keyword.validate!([strategies: nil] ++ @selection_defaults)
      |> validate_options!()

    strategies = validate_strategies!(Keyword.fetch!(opts, :strategies))
    rows = Enum.to_list(rows)
    calibration_opts = Keyword.delete(opts, :strategies)

    strategies
    |> Enum.with_index()
    |> Enum.map(fn {{strategy, probability_accessor}, index} ->
      result =
        calibrate(rows, Keyword.put(calibration_opts, :probability, probability_accessor))

      {Map.put(result, :strategy, strategy), index}
    end)
    |> Enum.max_by(fn {result, index} ->
      {result.score, -result.false_positive_rate, result.accuracy, -index}
    end)
    |> elem(0)
  end

  defp normalize_rows(rows, opts) do
    rows
    |> Enum.to_list()
    |> Enum.map(fn row ->
      label = row |> fetch_value!(Keyword.fetch!(opts, :label), :label) |> normalize_label!()

      %{
        probability:
          row
          |> fetch_value!(Keyword.fetch!(opts, :probability), :probability)
          |> validate_unit_value!(:probability),
        positive?: label,
        target: normalize_target(row, label, Keyword.fetch!(opts, :target)),
        eligible?: normalize_eligibility(row, Keyword.fetch!(opts, :eligible))
      }
    end)
  end

  defp normalize_target(_row, label, nil), do: if(label, do: 1.0, else: 0.0)

  defp normalize_target(row, _label, accessor) do
    row
    |> fetch_value!(accessor, :target)
    |> validate_unit_value!(:target)
  end

  defp normalize_eligibility(_row, nil), do: true

  defp normalize_eligibility(row, accessor) do
    row
    |> fetch_value!(accessor, :eligible)
    |> validate_eligible!()
  end

  defp eligible_rows!(rows) do
    eligible = Enum.filter(rows, & &1.eligible?)

    if eligible == [] do
      raise ArgumentError, "cannot evaluate because every row is ineligible"
    end

    eligible
  end

  defp metric_context(all_rows, eligible, opts) do
    positive_count = Enum.count(eligible, & &1.positive?)
    total = length(eligible)

    brier =
      eligible
      |> Enum.sum_by(&:math.pow(&1.probability - &1.target, 2))
      |> Kernel./(total)

    %{
      total: total,
      input_count: length(all_rows),
      excluded_count: length(all_rows) - total,
      positive_count: positive_count,
      negative_count: total - positive_count,
      brier: brier,
      objective: Keyword.fetch!(opts, :objective),
      false_positive_cost: Keyword.fetch!(opts, :false_positive_cost)
    }
  end

  defp count_predictions(rows, threshold) do
    Enum.reduce(
      rows,
      %{true_positives: 0, true_negatives: 0, false_positives: 0, false_negatives: 0},
      fn row, counts ->
        case {row.positive?, row.probability >= threshold} do
          {true, true} -> Map.update!(counts, :true_positives, &(&1 + 1))
          {false, false} -> Map.update!(counts, :true_negatives, &(&1 + 1))
          {false, true} -> Map.update!(counts, :false_positives, &(&1 + 1))
          {true, false} -> Map.update!(counts, :false_negatives, &(&1 + 1))
        end
      end
    )
  end

  defp mark_positive(%{positive?: true}, counts) do
    counts
    |> Map.update!(:true_positives, &(&1 + 1))
    |> Map.update!(:false_negatives, &(&1 - 1))
  end

  defp mark_positive(%{positive?: false}, counts) do
    counts
    |> Map.update!(:false_positives, &(&1 + 1))
    |> Map.update!(:true_negatives, &(&1 - 1))
  end

  defp metrics(threshold, counts, context) do
    accuracy = ratio(counts.true_positives + counts.true_negatives, context.total)
    precision = ratio(counts.true_positives, counts.true_positives + counts.false_positives)
    recall = ratio(counts.true_positives, context.positive_count)
    specificity = ratio(counts.true_negatives, context.negative_count)
    false_positive_rate = ratio(counts.false_positives, context.negative_count)
    balanced_accuracy = (recall + specificity) / 2
    f1 = ratio(2 * precision * recall, precision + recall)

    objective_value =
      case context.objective do
        :accuracy -> accuracy
        :balanced_accuracy -> balanced_accuracy
        :f1 -> f1
      end

    %{
      threshold: threshold,
      input_count: context.input_count,
      total: context.total,
      excluded_count: context.excluded_count,
      true_positives: counts.true_positives,
      true_negatives: counts.true_negatives,
      false_positives: counts.false_positives,
      false_negatives: counts.false_negatives,
      accuracy: accuracy,
      precision: precision,
      recall: recall,
      specificity: specificity,
      false_positive_rate: false_positive_rate,
      balanced_accuracy: balanced_accuracy,
      f1: f1,
      brier: context.brier,
      objective: context.objective,
      objective_value: objective_value,
      false_positive_cost: context.false_positive_cost,
      score: objective_value - context.false_positive_cost * false_positive_rate
    }
  end

  defp metric_order(metric) do
    {metric.score, -metric.false_positive_rate, metric.accuracy, metric.threshold}
  end

  defp validate_strategies!(strategies) when is_list(strategies) and strategies != [] do
    Enum.each(strategies, fn
      {_name, accessor} ->
        validate_accessor!(accessor, :strategies)

      strategy ->
        raise ArgumentError,
              ":strategies must contain {name, accessor} pairs, got: #{inspect(strategy)}"
    end)

    names = Enum.map(strategies, &elem(&1, 0))

    if Enum.uniq(names) != names do
      raise ArgumentError, ":strategies must use unique names"
    end

    strategies
  end

  defp validate_strategies!(strategies) do
    raise ArgumentError,
          ":strategies must be a non-empty list of {name, accessor} pairs, got: " <>
            inspect(strategies)
  end

  defp validate_options!(opts) do
    validate_accessor!(Keyword.fetch!(opts, :label), :label)

    if Keyword.has_key?(opts, :probability) do
      validate_accessor!(Keyword.fetch!(opts, :probability), :probability)
    end

    for option <- [:target, :eligible],
        accessor = Keyword.fetch!(opts, option),
        accessor != nil do
      validate_accessor!(accessor, option)
    end

    validate_objective!(Keyword.fetch!(opts, :objective))

    opts
    |> Keyword.fetch!(:false_positive_cost)
    |> validate_false_positive_cost!()

    opts
  end

  defp fetch_value!(row, accessor, _option) when is_function(accessor, 1), do: accessor.(row)

  defp fetch_value!(row, key, option) when is_atom(key) and is_map(row) do
    case Map.fetch(row, key) do
      {:ok, value} ->
        value

      :error ->
        raise ArgumentError, "#{inspect(option)} accessor could not find key #{inspect(key)}"
    end
  end

  defp fetch_value!(row, key, option) when is_atom(key) do
    raise ArgumentError,
          "#{inspect(option)} map-key accessor #{inspect(key)} requires a map row, got: " <>
            inspect(row)
  end

  defp fetch_value!(_row, accessor, option) do
    validate_accessor!(accessor, option)
  end

  defp validate_accessor!(accessor, _option)
       when is_atom(accessor) or is_function(accessor, 1),
       do: accessor

  defp validate_accessor!(accessor, option) do
    raise ArgumentError,
          "#{inspect(option)} must be a map key or one-argument function, got: " <>
            inspect(accessor)
  end

  defp normalize_label!(value) when value in [1, true], do: true
  defp normalize_label!(value) when value in [0, false], do: false

  defp normalize_label!(value) do
    raise ArgumentError, ":label must resolve to 0, 1, false, or true, got: #{inspect(value)}"
  end

  defp validate_eligible!(value) when is_boolean(value), do: value

  defp validate_eligible!(value) do
    raise ArgumentError, ":eligible must resolve to a boolean, got: #{inspect(value)}"
  end

  defp validate_unit_value!(value, _option)
       when is_number(value) and value >= 0.0 and value <= 1.0,
       do: value / 1

  defp validate_unit_value!(value, option) do
    raise ArgumentError,
          "#{inspect(option)} must resolve to a number from 0.0 through 1.0, got: " <>
            inspect(value)
  end

  defp validate_threshold!(threshold), do: validate_unit_value!(threshold, :threshold)

  defp validate_objective!(objective) when objective in @objectives, do: objective

  defp validate_objective!(objective) do
    raise ArgumentError,
          ":objective must be one of #{inspect(@objectives)}, got: #{inspect(objective)}"
  end

  defp validate_false_positive_cost!(cost) when is_number(cost) and cost >= 0, do: cost / 1

  defp validate_false_positive_cost!(cost) do
    raise ArgumentError,
          ":false_positive_cost must be a non-negative number, got: #{inspect(cost)}"
  end

  defp midpoint(upper, lower), do: lower + (upper - lower) / 2
  defp ratio(_numerator, denominator) when denominator == 0, do: 0.0
  defp ratio(numerator, denominator), do: numerator / denominator
end
