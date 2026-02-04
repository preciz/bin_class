# examples/configurable_backend.exs

# This example demonstrates how to use a custom compiler and definition options.
# While EXLA is the default, you might want to use others like Nx.BinaryBackend
# for environments where XLA is not available, or Torchx.

data = [
  %{text: "This is amazing!", label: 1},
  %{text: "I love it.", label: 1},
  %{text: "Not good.", label: 0},
  %{text: "Terrible experience.", label: 0}
]

data = List.duplicate(data, 10) |> List.flatten()

# 1. Training with a specific compiler (optional, defaults to EXLA)
# Here we use EXLA explicitly.
classifier =
  BinClass.Trainer.train(data,
    epochs: 5,
    labels: %{0 => :negative, 1 => :positive},
    compiler: EXLA
  )

model_path = "configurable_model.bin"
BinClass.save(classifier, model_path)

# 2. Loading with a specific compiler and defn_options
# This is useful for inference-time optimization or fallback backends.
serving = BinClass.load(model_path,
  compiler: EXLA,
  defn_options: [compiler: EXLA]
)

prediction = Nx.Serving.run(serving, "What a wonderful tool!")
IO.puts("Prediction: #{prediction.label} (Confidence: #{Float.round(prediction.confidence * 100, 2)}%)")
