defmodule BinClass.MixProject do
  use Mix.Project

  @version "0.1.0"
  @github "https://github.com/preciz/bin_class"

  def project do
    [
      app: :bin_class,
      version: @version,
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # Hex
      package: package(),
      description:
        "An easy-to-use library for building, training, and deploying binary text classifiers with Axon.",

      # Docs
      name: "BinClass",
      docs: docs()
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

  defp package do
    [
      maintainers: ["Barna Kovacs"],
      licenses: ["MIT"],
      links: %{"GitHub" => @github}
    ]
  end

  defp docs do
    [
      main: "BinClass",
      source_ref: "v#{@version}",
      source_url: @github,
      extras: ["README.md"]
    ]
  end
end
