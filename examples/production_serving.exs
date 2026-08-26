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

case result do
  %{label: :insufficient_input, active_token_count: active_token_count} ->
    IO.puts("#{text} -> insufficient input (#{active_token_count} active tokens)")

  %{label: label, confidence: confidence, active_token_count: active_token_count} ->
    percent = Float.round(confidence * 100, 2)
    IO.puts("#{text} -> #{label} (#{percent}%, #{active_token_count} active tokens)")
end
