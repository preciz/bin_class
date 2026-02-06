data = [
  {"This is a positive text example for training.", 1},
  {"This is a negative text example for training.", 0},
  {"Another great feature of this product.", 1},
  {"I really dislike the service here.", 0}
]

# Multiply to get enough data
large_data = 
  Stream.cycle(data) 
  |> Stream.map(fn {t, l} -> %{text: t, label: l} end)
  |> Enum.take(1000)

# Train a model
IO.puts("Training model for benchmarking...")
classifier = BinClass.Trainer.train(large_data, epochs: 2, batch_size: 32)
BinClass.save(classifier, "benchmark_predictor.bin")

# 1. Serving Benchmark
IO.puts("\nPreparing Nx.Serving...")
serving = BinClass.load("benchmark_predictor.bin")
sample_text = "This is a test for speed performance."

# Warmup
Enum.each(1..100, fn _ -> Nx.Serving.run(serving, sample_text) end)

IO.puts("Benchmarking Nx.Serving (1000 iter)...")
{serving_time, _} = :timer.tc(fn ->
  Enum.each(1..1000, fn _ -> Nx.Serving.run(serving, sample_text) end)
end)

# 2. Predictor Benchmark
IO.puts("\nPreparing Compiled Predictor...")
predictor = BinClass.compile_predictor(classifier)

# Warmup
Enum.each(1..100, fn _ -> predictor.(sample_text) end)

IO.puts("Benchmarking Compiled Predictor (1000 iter)...")
{predictor_time, _} = :timer.tc(fn ->
  Enum.each(1..1000, fn _ -> predictor.(sample_text) end)
end)

IO.puts("\n\n")
IO.puts("| Mode | Total Time (ms) | Avg Latency (ms) | Speedup |")
IO.puts("| :--- | :--- | :--- | :--- |")
IO.puts("| Nx.Serving | #{serving_time / 1000} | #{serving_time / 1000 / 1000} | 1x |")
IO.puts("| Predictor | #{predictor_time / 1000} | #{predictor_time / 1000 / 1000} | #{Float.round(serving_time / predictor_time, 2)}x |")

File.rm!("benchmark_predictor.bin")