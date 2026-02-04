defmodule BinClass.MixProject do
  use Mix.Project

  def project do
    [
      app: :bin_class,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # Docs
      name: "BinClass",
      docs: [
        main: "BinClass",
        extras: ["README.md"]
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:axon, "~> 0.8"},
      {:exla, "~> 0.10"},
      {:nx, "~> 0.10"},
      {:tokenizers, "~> 0.5"},
      {:explorer, "~> 0.11"},
      {:ex_doc, "~> 0.40", only: :dev, runtime: false}
    ]
  end
end
