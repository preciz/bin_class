# examples/configurable_backend.exs

# This example demonstrates how to select a custom Nx compiler. EXLA is the
# default; Nx.Defn.Evaluator is a portable fallback that does not use XLA.

data = [
  %{text: "This is amazing!", label: 1},
  %{text: "I love it.", label: 1},
  %{text: "Not good.", label: 0},
  %{text: "Terrible experience.", label: 0}
]

data = List.duplicate(data, 2) |> List.flatten()

# 1. Train with a specific compiler (optional; defaults to EXLA).
classifier =
  BinClass.Trainer.train(data,
    epochs: 1,
    batch_size: 2,
    vector_length: 16,
    labels: %{0 => "negative", 1 => "positive"},
    compiler: Nx.Defn.Evaluator,
    seed: 42
  )

model_path = "configurable_model.bin"
BinClass.save(classifier, model_path)

# 2. Select the same compiler when loading for inference.
serving = BinClass.load(model_path, compiler: Nx.Defn.Evaluator)

prediction = Nx.Serving.run(serving, "What a wonderful tool!")

case prediction do
  %{label: :insufficient_input, active_token_count: active_token_count} ->
    IO.puts("Prediction: insufficient input (#{active_token_count} active tokens)")

  %{label: label, confidence: confidence, active_token_count: active_token_count} ->
    percent = Float.round(confidence * 100, 2)
    IO.puts("Prediction: #{label} (#{percent}%, #{active_token_count} active tokens)")
end
