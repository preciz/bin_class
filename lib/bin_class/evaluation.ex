defmodule BinClass.Evaluation do
  @moduledoc """
  Evaluates positive-class probabilities and calibrates decision thresholds.

  Rows may have any shape. The `:probability`, `:label`, `:target`, and
  `:eligible` options accept either a map key or a one-argument accessor
  function. By default, rows are expected to contain `:probability` and
  `:label` keys. The label must be `0`, `1`, `false`, or `true`.

  A separate soft `:target` may be supplied for the Brier score. It defaults to
  the binary label. Rows are eligible by default; an `:eligible` accessor can
  exclude abstentions or otherwise invalid inputs from every metric.

  ## Fixed-threshold evaluation

      BinClass.Evaluation.evaluate(rows,
        threshold: 0.7,
        probability: :positive_probability,
        label: :positive?
      )

  ## Threshold calibration

      BinClass.Evaluation.calibrate(rows,
        objective: :balanced_accuracy,
        false_positive_cost: 0.5
      )

  Calibration evaluates every distinct decision partition represented by the
  observed probabilities. The returned threshold is centered between adjacent
  probabilities whenever possible.

  ## Strategy selection

      BinClass.Evaluation.select(rows,
        strategies: [
          maximum: fn row -> row.probabilities.maximum end,
          top_two_mean: fn row -> row.probabilities.top_two_mean end
        ],
        label: :label
      )

  Each strategy is calibrated independently. The result includes the selected
  strategy and its metrics.
  """

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
  Evaluates rows at a fixed positive-class threshold.

  In addition to confusion-matrix counts, the result contains accuracy,
  precision, recall, specificity, false-positive rate, balanced accuracy, F1,
  Brier score, and the configured objective score.

  ## Options

    * `:threshold` - positive-class threshold from `0.0` through `1.0`.
      Defaults to `0.5`.
    * `:probability` - map key or one-argument probability accessor. Defaults
      to `:probability`.
    * `:label` - map key or one-argument binary-label accessor. Defaults to
      `:label`.
    * `:target` - optional map key or accessor for a soft target from `0.0`
      through `1.0`, used for the Brier score. Defaults to the binary label.
    * `:eligible` - optional map key or accessor returning a boolean. Ineligible
      rows are counted in `:excluded_count` but omitted from metrics.
    * `:objective` - `:accuracy`, `:balanced_accuracy`, or `:f1`. Defaults to
      `:accuracy`.
    * `:false_positive_cost` - non-negative multiple of the false-positive rate
      subtracted from the objective. Defaults to `0.0`.
  """
  def evaluate(rows, opts \\ []) do
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
