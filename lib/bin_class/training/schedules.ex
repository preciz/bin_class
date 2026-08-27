defmodule BinClass.Training.Schedules do
  @moduledoc false

  import Nx.Defn

  @doc false
  def build(:constant, learning_rate, _total_steps) do
    Polaris.Schedules.constant(learning_rate)
  end

  def build({:constant, _opts}, learning_rate, _total_steps) do
    Polaris.Schedules.constant(learning_rate)
  end

  def build(:linear, learning_rate, total_steps) do
    build({:linear, []}, learning_rate, total_steps)
  end

  def build({:linear, opts}, learning_rate, total_steps) when is_list(opts) do
    opts = Keyword.validate!(opts, warmup_ratio: 0.05, end_learning_rate: 0.0)
    warmup_ratio = validate_ratio!(Keyword.fetch!(opts, :warmup_ratio), :warmup_ratio)
    end_lr = validate_end_lr!(Keyword.fetch!(opts, :end_learning_rate), learning_rate)
    warmup_steps = max(1, round(total_steps * warmup_ratio))

    &apply_linear_decay(&1, learning_rate, total_steps, warmup_steps, end_lr)
  end

  def build(:cosine, learning_rate, total_steps) do
    build({:cosine, []}, learning_rate, total_steps)
  end

  def build({:cosine, opts}, learning_rate, total_steps) when is_list(opts) do
    opts = Keyword.validate!(opts, warmup_ratio: 0.05, end_learning_rate: 0.0)
    warmup_ratio = validate_ratio!(Keyword.fetch!(opts, :warmup_ratio), :warmup_ratio)
    end_lr = validate_end_lr!(Keyword.fetch!(opts, :end_learning_rate), learning_rate)
    warmup_steps = max(1, round(total_steps * warmup_ratio))

    &apply_cosine_decay(&1, learning_rate, total_steps, warmup_steps, end_lr)
  end

  def build(:wsd, learning_rate, total_steps) do
    build({:wsd, []}, learning_rate, total_steps)
  end

  def build({:wsd, opts}, learning_rate, total_steps) when is_list(opts) do
    opts = Keyword.validate!(opts, warmup_ratio: 0.05, decay_ratio: 0.20, end_learning_rate: 0.0)
    warmup_ratio = validate_ratio!(Keyword.fetch!(opts, :warmup_ratio), :warmup_ratio)
    decay_ratio = validate_ratio!(Keyword.fetch!(opts, :decay_ratio), :decay_ratio)
    end_lr = validate_end_lr!(Keyword.fetch!(opts, :end_learning_rate), learning_rate)
    warmup_steps = max(1, round(total_steps * warmup_ratio))
    decay_steps = max(1, round(total_steps * decay_ratio))

    &apply_wsd(&1, learning_rate, total_steps, warmup_steps, decay_steps, end_lr)
  end

  def build(custom_schedule, _learning_rate, _total_steps) when is_function(custom_schedule, 1) do
    custom_schedule
  end

  def build(invalid, _learning_rate, _total_steps) do
    raise ArgumentError,
          ":schedule must be :constant, :linear, :cosine, :wsd, a tuple with options, " <>
            "or a 1-argument schedule function, got: #{inspect(invalid)}"
  end

  def validate_schedule!(schedule) when schedule in [:constant, :linear, :cosine, :wsd],
    do: schedule

  def validate_schedule!({schedule, opts} = tuple)
      when schedule in [:constant, :linear, :cosine, :wsd] and is_list(opts),
      do: tuple

  def validate_schedule!(schedule) when is_function(schedule, 1), do: schedule

  def validate_schedule!(schedule) do
    raise ArgumentError,
          ":schedule must be :constant, :linear, :cosine, :wsd, a tuple with options, " <>
            "or a 1-argument schedule function, got: #{inspect(schedule)}"
  end

  defnp apply_linear_decay(step, init_value, total_steps, warmup_steps, end_value) do
    factor =
      if step < warmup_steps do
        step / Nx.max(1, warmup_steps)
      else
        Nx.max(0.0, (total_steps - step) / Nx.max(1, total_steps - warmup_steps))
      end

    end_value + (init_value - end_value) * factor
  end

  defnp apply_cosine_decay(step, init_value, total_steps, warmup_steps, end_value) do
    if step < warmup_steps do
      step / Nx.max(1, warmup_steps) * init_value
    else
      t = Nx.min(step - warmup_steps, total_steps - warmup_steps)
      decay_steps = Nx.max(1, total_steps - warmup_steps)
      theta = t / decay_steps * Nx.Constants.pi()
      cos = (Nx.cos(theta) + 1) / 2
      end_value + (init_value - end_value) * cos
    end
  end

  defnp apply_wsd(step, init_value, total_steps, warmup_steps, decay_steps_count, end_value) do
    decay_start = total_steps - decay_steps_count

    if step < warmup_steps do
      step / Nx.max(1, warmup_steps) * init_value
    else
      if step < decay_start do
        init_value
      else
        decay_progress = Nx.min(1.0, (step - decay_start) / Nx.max(1, decay_steps_count))
        init_value - (init_value - end_value) * decay_progress
      end
    end
  end

  defp validate_ratio!(ratio, _name) when is_number(ratio) and ratio >= 0.0 and ratio <= 1.0,
    do: ratio / 1.0

  defp validate_ratio!(ratio, name) do
    raise ArgumentError,
          ":#{name} must be a number between 0.0 and 1.0, got: #{inspect(ratio)}"
  end

  defp validate_end_lr!(end_lr, init_lr)
       when is_number(end_lr) and end_lr >= 0.0 and end_lr <= init_lr,
       do: end_lr / 1.0

  defp validate_end_lr!(end_lr, init_lr) do
    raise ArgumentError,
          ":end_learning_rate must be a non-negative number <= initial learning rate (#{init_lr}), got: #{inspect(end_lr)}"
  end
end
