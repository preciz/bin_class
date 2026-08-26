defmodule BinClass.DecisionPolicyTest do
  use ExUnit.Case, async: true

  alias BinClass.DecisionPolicy

  test "normalizes supported training configurations" do
    assert DecisionPolicy.training_config!(:argmax) ==
             {:fixed, %{strategy: :argmax}}

    assert DecisionPolicy.training_config!({:positive_threshold, 0.7}) ==
             {:fixed, %{strategy: :positive_threshold, positive_threshold: 0.7}}

    assert DecisionPolicy.training_config!(:optimize_threshold) ==
             {:optimize, %{objective: :accuracy, false_positive_cost: 0.0}}

    assert DecisionPolicy.training_config!(
             {:optimize_threshold, [objective: :f1, false_positive_cost: 2.0]}
           ) == {:optimize, %{objective: :f1, false_positive_cost: 2.0}}
  end

  test "validates optimization configuration" do
    assert_raise ArgumentError, ~r/:objective must be one of/, fn ->
      DecisionPolicy.training_config!({:optimize_threshold, objective: :precision})
    end

    assert_raise ArgumentError, ~r/:false_positive_cost must be a non-negative number/, fn ->
      DecisionPolicy.training_config!({:optimize_threshold, false_positive_cost: -1})
    end

    assert_raise ArgumentError, ~r/unknown keys/, fn ->
      DecisionPolicy.training_config!({:optimize_threshold, recall_weight: 2})
    end
  end

  test "argmax and positive-threshold strategies retain distinct tie behavior" do
    assert DecisionPolicy.class_index([0.5, 0.5], %{strategy: :argmax}) == 0

    assert DecisionPolicy.class_index(
             [0.5, 0.5],
             %{strategy: :positive_threshold, positive_threshold: 0.5}
           ) == 1
  end

  test "validates persisted policies and minimum input length" do
    policy = %{
      strategy: :positive_threshold,
      positive_threshold: 0.6,
      optimization: %{objective: :accuracy, false_positive_cost: 0.5}
    }

    assert DecisionPolicy.validate!(policy) == policy
    assert DecisionPolicy.validate_minimum_input_tokens!(8, 8) == 8

    assert_raise ArgumentError, ~r/invalid threshold optimization metadata/, fn ->
      DecisionPolicy.validate!(%{policy | optimization: %{objective: :precision}})
    end

    assert_raise ArgumentError, ~r/cannot exceed :vector_length/, fn ->
      DecisionPolicy.validate_minimum_input_tokens!(9, 8)
    end

    assert_raise ArgumentError, ~r/invalid decision policy/, fn ->
      DecisionPolicy.validate!(Map.put(policy, :unexpected, true))
    end

    assert_raise ArgumentError, ~r/invalid decision policy/, fn ->
      DecisionPolicy.validate!(%{strategy: :automatic})
    end
  end
end
