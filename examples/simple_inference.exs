# examples/simple_inference.exs

model_path = "movie_classifier.bin"

unless File.exists?(model_path) do
  IO.puts("Error: #{model_path} not found. Please run train_and_save.exs first.")
  System.halt(1)
end

serving = BinClass.load(model_path)

# Single prediction
text = "That was a great movie!"
prediction = Nx.Serving.run(serving, text)

IO.puts("#{text} -> #{prediction.label} (#{Float.round(prediction.confidence * 100, 2)}%)")

# Batch prediction
batch = [
  "It was okay, but a bit slow.",
  "Total disaster, I hated it."
]

results = Nx.Serving.run(serving, batch)

for {t, p} <- Enum.zip(batch, results) do
  IO.puts("#{t} -> #{p.label} (#{Float.round(p.confidence * 100, 2)}%)")
end