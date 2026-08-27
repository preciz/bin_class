defmodule BinClass.Training.Config do
  @moduledoc false

  alias BinClass.{DecisionPolicy, Labels, Model, Options}
  alias BinClass.Training.Schedules

  @default_architecture :cnn
  @default_vector_length 512

  @enforce_keys [
    :epochs,
    :batch_size,
    :learning_rate,
    :decay,
    :schedule,
    :labels,
    :target_mode,
    :target_threshold,
    :balance,
    :validation_split,
    :patience,
    :compiler,
    :architecture,
    :dropout_rate,
    :decision_config,
    :minimum_input_tokens,
    :vector_length,
    :seed
  ]

  defstruct @enforce_keys

  @doc false
  def new!(opts) do
    opts =
      Keyword.validate!(opts, [
        :epochs,
        :batch_size,
        :learning_rate,
        :decay,
        :schedule,
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
    seed = Keyword.get(opts, :seed)

    validate_target_options!(target_mode, target_threshold)
    Options.validate_optional_integer!(seed, :seed)

    architecture = Keyword.get(opts, :architecture, @default_architecture)
    Model.validate_name!(architecture)

    config = %__MODULE__{
      epochs: Keyword.get(opts, :epochs, 10),
      batch_size: Keyword.get(opts, :batch_size, 32),
      learning_rate:
        opts
        |> Keyword.get(:learning_rate, 1.0e-3)
        |> Options.validate_positive_number!(:learning_rate),
      decay:
        opts
        |> Keyword.get(:decay, 1.0e-2)
        |> Options.validate_non_negative_number!(:decay),
      schedule:
        opts
        |> Keyword.get(:schedule, :linear)
        |> Schedules.validate_schedule!(),
      labels: opts |> Keyword.get(:labels, Labels.default()) |> Labels.validate!(),
      target_mode: target_mode,
      target_threshold: target_threshold,
      balance: opts |> Keyword.get(:balance, true) |> Options.validate_boolean!(:balance),
      validation_split: Keyword.get(opts, :validation_split, 0.1),
      patience:
        opts
        |> Keyword.get(:patience, 5)
        |> Options.validate_non_negative_integer!(:patience),
      compiler: opts |> Keyword.get(:compiler, EXLA) |> Options.validate_compiler!(),
      architecture: architecture,
      dropout_rate:
        opts
        |> Keyword.get(:dropout_rate, 0.2)
        |> Options.validate_dropout_rate!(),
      decision_config:
        opts
        |> Keyword.get(:decision_policy, :argmax)
        |> DecisionPolicy.training_config!(),
      minimum_input_tokens:
        opts
        |> Keyword.get(:minimum_input_tokens, 0)
        |> DecisionPolicy.validate_minimum_input_tokens!(),
      vector_length: Keyword.get(opts, :vector_length, @default_vector_length),
      seed: seed
    }

    validate_configuration!(config)
  end

  defp validate_target_options!(target_mode, target_threshold) do
    if target_mode not in [:hard, :soft] do
      raise ArgumentError,
            ":target_mode must be :hard or :soft, got: #{inspect(target_mode)}"
    end

    if !(is_number(target_threshold) and target_threshold >= 0 and target_threshold <= 1) do
      raise ArgumentError,
            ":target_threshold must be a number between 0 and 1, got: " <>
              inspect(target_threshold)
    end
  end

  defp validate_configuration!(config) do
    if !(is_integer(config.epochs) and config.epochs > 0) do
      raise ArgumentError,
            ":epochs must be a positive integer, got: #{inspect(config.epochs)}"
    end

    if !(is_integer(config.batch_size) and config.batch_size > 0) do
      raise ArgumentError,
            ":batch_size must be a positive integer, got: #{inspect(config.batch_size)}"
    end

    if !(is_number(config.validation_split) and config.validation_split > 0 and
           config.validation_split < 1) do
      raise ArgumentError,
            ":validation_split must be a number greater than 0 and less than 1, " <>
              "got: #{inspect(config.validation_split)}"
    end

    Model.validate_vector_length!(config.architecture, config.vector_length)

    if config.minimum_input_tokens > config.vector_length do
      raise ArgumentError,
            ":minimum_input_tokens cannot exceed :vector_length, got " <>
              "#{config.minimum_input_tokens} and #{config.vector_length}"
    end

    config
  end
end
