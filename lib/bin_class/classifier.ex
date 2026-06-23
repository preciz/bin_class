defmodule BinClass.Classifier do
  @moduledoc """
  A struct representing a trained binary classifier.

  The optional `:decision_policy` field stores calibrated inference settings
  such as `:positive_threshold` and `:min_positive_tokens`. It is persisted
  with newly serialized classifiers and may be `nil` for older saved models.
  """

  defstruct [
    :model_params,
    :tokenizer,
    :labels,
    :vector_length,
    :vocab_size,
    :accuracy,
    :epoch,
    :model_version,
    :learning_rate,
    :dropout_rate,
    :decision_policy
  ]

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(classifier, opts) do
      concat([
        "#BinClass.Classifier<",
        to_doc(
          %{
            model_version: classifier.model_version,
            labels: classifier.labels,
            vector_length: classifier.vector_length,
            vocab_size: classifier.vocab_size,
            accuracy: classifier.accuracy,
            epoch: classifier.epoch,
            learning_rate: classifier.learning_rate,
            dropout_rate: classifier.dropout_rate,
            decision_policy: classifier.decision_policy
          },
          opts
        ),
        ">"
      ])
    end
  end
end
