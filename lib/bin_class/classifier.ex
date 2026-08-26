defmodule BinClass.Classifier do
  @moduledoc """
  A struct representing a trained binary classifier.

  The required `:decision_policy` field controls how probabilities become a
  class label. It is either `%{strategy: :argmax}` or a positive-threshold
  policy. `:minimum_input_tokens` is a separate input-validity requirement;
  inputs below it produce `:insufficient_input` without hiding probabilities.

  `:target_mode` and `:target_threshold` record whether hard or soft targets
  were used during training and how soft targets were binarized for metrics.
  They do not change inference behavior.

  `:model_revision` identifies the exact graph semantics used for training.
  It prevents classifiers trained with older masking behavior from silently
  running against a different graph.

  `:vocab_size`, `:pad_token_id`, and `:unk_token_id` are derived from the
  trained tokenizer and validated when the classifier is serialized or loaded.
  Persisted output labels are exactly two distinct binaries. Serialization and
  inference require every field in the current struct schema; the struct does
  not supply compatibility defaults for manually assembled classifiers.

  """

  alias BinClass.{DecisionPolicy, Labels, Model, Tokenizer}

  defstruct [
    :model_params,
    :tokenizer,
    :labels,
    :vector_length,
    :vocab_size,
    :pad_token_id,
    :unk_token_id,
    :accuracy,
    :epoch,
    :architecture,
    :learning_rate,
    :dropout_rate,
    :decision_policy,
    :minimum_input_tokens,
    :target_mode,
    :target_threshold,
    :model_revision
  ]

  @doc false
  def validate!(%__MODULE__{} = classifier) do
    Model.validate_name!(classifier.architecture)
    Model.validate_revision!(classifier.model_revision)
    Model.validate_vector_length!(classifier.architecture, classifier.vector_length)
    Labels.validate!(classifier.labels)
    DecisionPolicy.validate!(classifier.decision_policy)
    validate_dropout_rate!(classifier.dropout_rate)
    validate_accuracy!(classifier.accuracy)
    validate_epoch!(classifier.epoch)
    validate_learning_rate!(classifier.learning_rate)
    validate_target_mode!(classifier.target_mode)
    validate_target_threshold!(classifier.target_threshold)

    DecisionPolicy.validate_minimum_input_tokens!(
      classifier.minimum_input_tokens,
      classifier.vector_length
    )

    Tokenizer.validate_metadata!(classifier.tokenizer, %{
      vocab_size: classifier.vocab_size,
      pad_token_id: classifier.pad_token_id,
      unk_token_id: classifier.unk_token_id
    })

    classifier
  end

  defp validate_accuracy!(accuracy)
       when is_number(accuracy) and accuracy >= 0 and accuracy <= 1,
       do: accuracy

  defp validate_accuracy!(accuracy) do
    raise ArgumentError,
          "Classifier :accuracy must be a number between 0 and 1, got: #{inspect(accuracy)}"
  end

  defp validate_epoch!(epoch) when is_integer(epoch) and epoch >= 0, do: epoch

  defp validate_epoch!(epoch) do
    raise ArgumentError,
          "Classifier :epoch must be a non-negative integer, got: #{inspect(epoch)}"
  end

  defp validate_learning_rate!(learning_rate)
       when is_number(learning_rate) and learning_rate > 0,
       do: learning_rate

  defp validate_learning_rate!(learning_rate) do
    raise ArgumentError,
          "Classifier :learning_rate must be a positive number, got: #{inspect(learning_rate)}"
  end

  defp validate_dropout_rate!(dropout_rate)
       when is_number(dropout_rate) and dropout_rate >= 0 and dropout_rate < 1,
       do: dropout_rate

  defp validate_dropout_rate!(dropout_rate) do
    raise ArgumentError,
          "Classifier :dropout_rate must be a number greater than or equal to 0 and below 1, " <>
            "got: #{inspect(dropout_rate)}"
  end

  defp validate_target_mode!(target_mode) when target_mode in [:hard, :soft], do: target_mode

  defp validate_target_mode!(target_mode) do
    raise ArgumentError,
          "Classifier :target_mode must be :hard or :soft, got: #{inspect(target_mode)}"
  end

  defp validate_target_threshold!(target_threshold)
       when is_number(target_threshold) and target_threshold >= 0 and target_threshold <= 1,
       do: target_threshold

  defp validate_target_threshold!(target_threshold) do
    raise ArgumentError,
          "Classifier :target_threshold must be a number between 0 and 1, " <>
            "got: #{inspect(target_threshold)}"
  end

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(classifier, opts) do
      concat([
        "#BinClass.Classifier<",
        to_doc(
          %{
            architecture: classifier.architecture,
            model_revision: classifier.model_revision,
            labels: classifier.labels,
            vector_length: classifier.vector_length,
            vocab_size: classifier.vocab_size,
            pad_token_id: classifier.pad_token_id,
            unk_token_id: classifier.unk_token_id,
            accuracy: classifier.accuracy,
            epoch: classifier.epoch,
            learning_rate: classifier.learning_rate,
            dropout_rate: classifier.dropout_rate,
            decision_policy: classifier.decision_policy,
            minimum_input_tokens: classifier.minimum_input_tokens,
            target_mode: classifier.target_mode,
            target_threshold: classifier.target_threshold
          },
          opts
        ),
        ">"
      ])
    end
  end
end
