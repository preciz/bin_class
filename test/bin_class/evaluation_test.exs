defmodule BinClass.EvaluationTest do
  use ExUnit.Case, async: true

  alias BinClass.Evaluation

  test "evaluates a fixed threshold with binary labels and soft Brier targets" do
    rows = [
      %{probability: 0.9, label: 1, target: 0.8},
      %{probability: 0.7, label: 0, target: 0.4},
      %{probability: 0.4, label: 1, target: 0.6},
      %{probability: 0.1, label: 0, target: 0.0}
    ]

    result = Evaluation.evaluate(rows, threshold: 0.6, target: :target)

    assert result.total == 4
    assert result.input_count == 4
    assert result.excluded_count == 0
    assert result.true_positives == 1
    assert result.true_negatives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.accuracy == 0.5
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.specificity == 0.5
    assert result.false_positive_rate == 0.5
    assert result.balanced_accuracy == 0.5
    assert result.f1 == 0.5
    assert_in_delta result.brier, 0.0375, 1.0e-12
    assert result.objective == :accuracy
    assert result.objective_value == 0.5
    assert result.score == 0.5
  end

  test "uses custom accessors and excludes ineligible rows from every metric" do
    rows = [
      %{scores: %{document: 0.9}, class: true, usable?: true},
      %{scores: %{document: 0.2}, class: false, usable?: true},
      %{scores: %{document: 1.0}, class: false, usable?: false}
    ]

    result =
      Evaluation.evaluate(rows,
        threshold: 0.5,
        probability: & &1.scores.document,
        label: :class,
        eligible: :usable?
      )

    assert result.input_count == 3
    assert result.total == 2
    assert result.excluded_count == 1
    assert result.accuracy == 1.0
    assert_in_delta result.brier, 0.025, 1.0e-12
  end

  test "calibrates at exact observed probability boundaries" do
    rows = [
      %{probability: 0.99, eligible: false, label: 1},
      %{probability: 0.83, eligible: true, label: 1},
      %{probability: 0.72, eligible: true, label: 0},
      %{probability: 0.61, eligible: true, label: 1},
      %{probability: 0.40, eligible: true, label: 0}
    ]

    result =
      Evaluation.calibrate(rows,
        eligible: :eligible,
        objective: :accuracy,
        false_positive_cost: 0.75
      )

    assert_in_delta result.threshold, 0.775, 1.0e-12
    assert result.accuracy == 0.75
    assert result.false_positive_rate == 0.0
    assert result.score == 0.75
    assert result.excluded_count == 1
  end

  test "calibrates when the highest observed probability is exactly one" do
    result =
      Evaluation.calibrate([
        %{probability: 1.0, label: 1},
        %{probability: 0.0, label: 0}
      ])

    assert result.threshold == 0.5
    assert result.accuracy == 1.0
  end

  test "selects and identifies the best named probability strategy" do
    rows = [
      %{label: 1, scores: %{maximum: 0.9, mean: 0.9}},
      %{label: 1, scores: %{maximum: 0.8, mean: 0.8}},
      %{label: 0, scores: %{maximum: 0.85, mean: 0.3}},
      %{label: 0, scores: %{maximum: 0.7, mean: 0.2}}
    ]

    result =
      Evaluation.select(rows,
        strategies: [
          maximum: & &1.scores.maximum,
          mean: & &1.scores.mean
        ],
        false_positive_cost: 0.5
      )

    assert result.strategy == :mean
    assert result.accuracy == 1.0
    assert result.false_positive_rate == 0.0
    assert_in_delta result.threshold, 0.55, 1.0e-12
  end

  test "selects the first strategy when outcomes are equivalent" do
    rows = [
      %{label: 1, first: 0.4, second: 0.8},
      %{label: 0, first: 0.2, second: 0.6}
    ]

    result =
      Evaluation.select(rows,
        strategies: [first: :first, second: :second]
      )

    assert result.strategy == :first
    assert result.accuracy == 1.0
    assert_in_delta result.threshold, 0.3, 1.0e-12
  end

  test "breaks equivalent outcomes toward lower FPR and a higher threshold" do
    result =
      Evaluation.calibrate(
        [
          %{probability: 0.8, label: 1},
          %{probability: 0.7, label: 1},
          %{probability: 0.1, label: 0}
        ],
        false_positive_cost: 0.5
      )

    assert_in_delta result.threshold, 0.4, 1.0e-12
    assert result.accuracy == 1.0
  end

  test "supports alternative objectives and explicit false-positive cost" do
    rows = [
      %{probability: 0.9, label: 0},
      %{probability: 0.8, label: 1},
      %{probability: 0.7, label: 1},
      %{probability: 0.1, label: 0}
    ]

    accuracy = Evaluation.calibrate(rows, objective: :accuracy)

    false_positive_averse =
      Evaluation.calibrate(rows, objective: :f1, false_positive_cost: 2.0)

    assert accuracy.threshold < 0.9
    assert false_positive_averse.threshold > 0.9
    assert false_positive_averse.false_positive_rate == 0.0
  end

  test "requires eligible rows and both classes for calibration" do
    assert_raise ArgumentError, ~r/every row is ineligible/, fn ->
      Evaluation.calibrate(
        [
          %{probability: 0.9, eligible: false, label: 1},
          %{probability: 0.1, eligible: false, label: 0}
        ],
        eligible: :eligible
      )
    end

    assert_raise ArgumentError, ~r/examples from both classes/, fn ->
      Evaluation.calibrate([
        %{probability: 0.9, label: 1},
        %{probability: 0.8, label: 1}
      ])
    end
  end

  test "validates rows and options directly" do
    assert_raise ArgumentError, ~r/unknown keys.*:not_an_option/s, fn ->
      Evaluation.evaluate([], not_an_option: true)
    end

    assert_raise ArgumentError, ~r/:probability must resolve/, fn ->
      Evaluation.evaluate([%{probability: 1.1, label: 1}])
    end

    assert_raise ArgumentError, ~r/:label must resolve/, fn ->
      Evaluation.evaluate([%{probability: 0.5, label: :positive}])
    end

    assert_raise ArgumentError, ~r/:objective must be one of/, fn ->
      Evaluation.evaluate([%{probability: 0.5, label: 1}], objective: :cost)
    end

    assert_raise ArgumentError, ~r/:strategies must be a non-empty list/, fn ->
      Evaluation.select([%{probability: 0.5, label: 1}], strategies: [])
    end
  end

  test "reports malformed strategies, accessors, eligibility, and costs" do
    rows = [%{probability: 0.5, label: 1}]

    assert_raise ArgumentError, ~r/must contain \{name, accessor\} pairs/, fn ->
      Evaluation.select(rows, strategies: [:not_a_pair])
    end

    assert_raise ArgumentError, ~r/must use unique names/, fn ->
      Evaluation.select(rows, strategies: [same: :probability, same: :probability])
    end

    assert_raise ArgumentError, ~r/could not find key :missing/, fn ->
      Evaluation.evaluate(rows, probability: :missing)
    end

    assert_raise ArgumentError, ~r/map-key accessor :probability requires a map row/, fn ->
      Evaluation.evaluate([{0.5, 1}], label: &elem(&1, 1))
    end

    assert_raise ArgumentError, ~r/:probability must be a map key or one-argument function/, fn ->
      Evaluation.evaluate(rows, probability: "probability")
    end

    assert_raise ArgumentError, ~r/:eligible must resolve to a boolean/, fn ->
      Evaluation.evaluate([%{probability: 0.5, label: 1, eligible: 1}], eligible: :eligible)
    end

    assert_raise ArgumentError, ~r/:false_positive_cost must be a non-negative number/, fn ->
      Evaluation.evaluate(rows, false_positive_cost: -1)
    end
  end
end
