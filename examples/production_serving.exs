# examples/production_serving.exs

model_path = "movie_classifier.bin"

unless File.exists?(model_path) do
  IO.puts("Error: #{model_path} not found. Please run train_and_save.exs first.")
  System.halt(1)
end

defmodule MyApp.Application do
  use Application

  def start(_type, _args) do
    model_path = "movie_classifier.bin"
    serving = BinClass.load(model_path)

    children = [
      {Nx.Serving, serving: serving, name: MyApp.Classifier, batch_timeout: 100}
    ]

    Supervisor.start_link(children, strategy: :one_for_one)
  end
end

{:ok, _} = MyApp.Application.start(:normal, [])

text = "The masterpiece of the decade!"
result = Nx.Serving.batched_run(MyApp.Classifier, text)

IO.puts("#{text} -> #{result.label} (#{Float.round(result.confidence * 100, 2)}%)")