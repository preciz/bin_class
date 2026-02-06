defmodule BinClass.Classifier do
  @moduledoc """
  A struct representing a trained binary classifier.
  """

  defstruct [
    :model_params,
    :tokenizer,
    :labels,
    :vector_length,
    :vocab_size,
    :accuracy,
    :epoch,
    :model_version
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
            epoch: classifier.epoch
          },
          opts
        ),
        ">"
      ])
    end
  end
end
