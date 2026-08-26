# examples/simple_inference.exs

model_path = "movie_classifier.bin"

unless File.exists?(model_path) do
  IO.puts("Error: #{model_path} not found. Please run train_and_save.exs first.")
  System.halt(1)
end

serving = BinClass.load(model_path)

format_prediction = fn text, prediction ->
  case prediction do
    %{label: :insufficient_input, active_token_count: active_token_count} ->
      "#{text} -> insufficient input (#{active_token_count} active tokens)"

    %{label: label, confidence: confidence, active_token_count: active_token_count} ->
      percent = Float.round(confidence * 100, 2)
      "#{text} -> #{label} (#{percent}%, #{active_token_count} active tokens)"
  end
end

# Single prediction
text = "That was a great movie!"
prediction = Nx.Serving.run(serving, text)
IO.puts(format_prediction.(text, prediction))

# Batch prediction
batch = [
  "It was okay, but a bit slow.",
  "Total disaster, I hated it."
]

results = Nx.Serving.run(serving, batch)

for {t, p} <- Enum.zip(batch, results) do
  IO.puts(format_prediction.(t, p))
end
