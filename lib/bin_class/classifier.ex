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
  Persisted output labels are exactly two distinct binaries.

  """

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
    model_revision: BinClass.Model.revision()
  ]

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
