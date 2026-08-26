defmodule BinClass.DecisionPolicy do
  @moduledoc false

  @objectives [:accuracy, :balanced_accuracy, :f1]

  def training_config!(:argmax), do: {:fixed, %{strategy: :argmax}}

  def training_config!({:positive_threshold, threshold}) do
    {:fixed,
     %{
       strategy: :positive_threshold,
       positive_threshold: validate_threshold!(threshold)
     }}
  end

  def training_config!(:optimize_threshold) do
    training_config!({:optimize_threshold, []})
  end

  def training_config!({:optimize_threshold, opts}) when is_list(opts) do
    opts = Keyword.validate!(opts, objective: :accuracy, false_positive_cost: 0.0)
    objective = Keyword.fetch!(opts, :objective)
    false_positive_cost = Keyword.fetch!(opts, :false_positive_cost)

    unless objective in @objectives do
      raise ArgumentError,
            "threshold optimization :objective must be one of #{inspect(@objectives)}, " <>
              "got: #{inspect(objective)}"
    end

    unless is_number(false_positive_cost) and false_positive_cost >= 0 do
      raise ArgumentError,
            "threshold optimization :false_positive_cost must be a non-negative number, " <>
              "got: #{inspect(false_positive_cost)}"
    end

    {:optimize,
     %{
       objective: objective,
       false_positive_cost: false_positive_cost
     }}
  end

  def training_config!(value) do
    raise ArgumentError,
          ":decision_policy must be :argmax, {:positive_threshold, threshold}, " <>
            ":optimize_threshold, or {:optimize_threshold, options}, got: #{inspect(value)}"
  end

  def validate!(%{strategy: :argmax} = policy) when map_size(policy) == 1, do: policy

  def validate!(%{strategy: :positive_threshold, positive_threshold: threshold} = policy) do
    validate_threshold!(threshold)

    unless Map.keys(policy)
           |> Enum.all?(&(&1 in [:strategy, :positive_threshold, :optimization])) do
      raise ArgumentError, "invalid decision policy: #{inspect(policy)}"
    end

    case Map.get(policy, :optimization) do
      nil -> :ok
      optimization -> validate_optimization_metadata!(optimization)
    end

    policy
  end

  def validate!(policy) do
    raise ArgumentError, "invalid decision policy: #{inspect(policy)}"
  end

  def validate_minimum_input_tokens!(value) when is_integer(value) and value >= 0, do: value

  def validate_minimum_input_tokens!(value) do
    raise ArgumentError,
          ":minimum_input_tokens must be a non-negative integer, got: #{inspect(value)}"
  end

  def validate_minimum_input_tokens!(value, vector_length) do
    value = validate_minimum_input_tokens!(value)

    if value > vector_length do
      raise ArgumentError,
            ":minimum_input_tokens cannot exceed :vector_length, got " <>
              "#{value} and #{vector_length}"
    end

    value
  end

  def class_index(probs, %{strategy: :argmax}) do
    max_probability = Enum.max(probs)
    Enum.find_index(probs, &(&1 == max_probability))
  end

  def class_index(probs, %{
        strategy: :positive_threshold,
        positive_threshold: threshold
      }) do
    if Enum.at(probs, 1) >= threshold, do: 1, else: 0
  end

  defp validate_threshold!(threshold)
       when is_number(threshold) and threshold >= 0 and threshold <= 1,
       do: threshold

  defp validate_threshold!(threshold) do
    raise ArgumentError,
          "positive threshold must be a number between 0 and 1, got: #{inspect(threshold)}"
  end

  defp validate_optimization_metadata!(%{
         objective: objective,
         false_positive_cost: false_positive_cost
       })
       when objective in @objectives and is_number(false_positive_cost) and
              false_positive_cost >= 0,
       do: :ok

  defp validate_optimization_metadata!(metadata) do
    raise ArgumentError, "invalid threshold optimization metadata: #{inspect(metadata)}"
  end
end
