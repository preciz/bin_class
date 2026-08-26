defmodule BinClass.EvaluationPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias BinClass.Evaluation

  @count_fields [
    :true_positives,
    :true_negatives,
    :false_positives,
    :false_negatives
  ]

  @bounded_metric_fields [
    :accuracy,
    :precision,
    :recall,
    :specificity,
    :false_positive_rate,
    :balanced_accuracy,
    :f1,
    :brier,
    :objective_value
  ]

  property "fixed-threshold metrics obey confusion-matrix invariants" do
    check all(
            rows <- list_of(row(), min_length: 1, max_length: 40),
            threshold <- unit_value(),
            max_runs: 200
          ) do
      result = Evaluation.evaluate(rows, threshold: threshold, eligible: :eligible)
      positive_count = Enum.count(rows, &(&1.label == 1))
      negative_count = length(rows) - positive_count

      assert result.total == length(rows)
      assert result.input_count == length(rows)
      assert result.excluded_count == 0
      assert result.true_positives + result.false_negatives == positive_count
      assert result.true_negatives + result.false_positives == negative_count
      assert Enum.sum(Map.take(result, @count_fields) |> Map.values()) == result.total

      Enum.each(@bounded_metric_fields, fn field ->
        value = Map.fetch!(result, field)
        assert value >= 0.0 and value <= 1.0
      end)

      reversed =
        Evaluation.evaluate(Enum.reverse(rows), threshold: threshold, eligible: :eligible)

      assert Map.take(reversed, @count_fields) == Map.take(result, @count_fields)

      Enum.each(@bounded_metric_fields, fn field ->
        assert_in_delta Map.fetch!(reversed, field), Map.fetch!(result, field), 1.0e-12
      end)

      excluded = %{probability: 1.0 - threshold, label: 1, eligible: false}

      with_excluded =
        Evaluation.evaluate(rows ++ [excluded], threshold: threshold, eligible: :eligible)

      assert with_excluded.input_count == result.input_count + 1
      assert with_excluded.excluded_count == 1

      assert Map.drop(with_excluded, [:input_count, :excluded_count]) ==
               Map.drop(result, [:input_count, :excluded_count])
    end
  end

  property "calibration matches exhaustive threshold evaluation" do
    check all(
            rows <- rows_with_both_classes(),
            objective <- member_of([:accuracy, :balanced_accuracy, :f1]),
            false_positive_cost <- integer(0..20) |> map(&(&1 / 10)),
            max_runs: 200
          ) do
      opts = [objective: objective, false_positive_cost: false_positive_cost]

      expected =
        rows
        |> candidate_thresholds()
        |> Enum.map(&Evaluation.evaluate(rows, Keyword.put(opts, :threshold, &1)))
        |> Enum.max_by(fn metrics ->
          {
            metrics.score,
            -metrics.false_positive_rate,
            metrics.accuracy,
            metrics.threshold
          }
        end)

      assert_calibration_equivalent(Evaluation.calibrate(rows, opts), expected)
      assert_calibration_equivalent(Evaluation.calibrate(Enum.reverse(rows), opts), expected)
    end
  end

  defp row do
    gen all(
          probability <- unit_value(),
          label <- member_of([0, 1])
        ) do
      %{probability: probability, label: label, eligible: true}
    end
  end

  defp rows_with_both_classes do
    gen all(
          positive_probabilities <- list_of(unit_value(), min_length: 1, max_length: 12),
          negative_probabilities <- list_of(unit_value(), min_length: 1, max_length: 12)
        ) do
      Enum.map(positive_probabilities, &%{probability: &1, label: 1}) ++
        Enum.map(negative_probabilities, &%{probability: &1, label: 0})
    end
  end

  defp unit_value do
    integer(0..20) |> map(&(&1 / 20))
  end

  defp candidate_thresholds(rows) do
    probabilities =
      rows
      |> Enum.map(& &1.probability)
      |> Enum.uniq()
      |> Enum.sort(:desc)

    thresholds =
      probabilities
      |> Enum.zip(Enum.drop(probabilities, 1) ++ [0.0])
      |> Enum.map(fn {upper, lower} -> lower + (upper - lower) / 2 end)

    case probabilities do
      [1.0 | _probabilities] -> thresholds
      [highest | _probabilities] -> [(1.0 + highest) / 2 | thresholds]
    end
  end

  defp assert_calibration_equivalent(actual, expected) do
    exact_fields = [
      :input_count,
      :total,
      :excluded_count,
      :true_positives,
      :true_negatives,
      :false_positives,
      :false_negatives,
      :objective,
      :false_positive_cost
    ]

    numeric_fields = [
      :threshold,
      :accuracy,
      :precision,
      :recall,
      :specificity,
      :false_positive_rate,
      :balanced_accuracy,
      :f1,
      :brier,
      :objective_value,
      :score
    ]

    assert Map.take(actual, exact_fields) == Map.take(expected, exact_fields)

    Enum.each(numeric_fields, fn field ->
      assert_in_delta Map.fetch!(actual, field), Map.fetch!(expected, field), 1.0e-12
    end)
  end
end
