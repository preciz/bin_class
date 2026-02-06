defmodule BinClass.Model do
  @moduledoc """
  Dispatcher for model architectures.

  This module handles model versioning, ensuring that trained models can be
  loaded even if the library's default architecture changes.

  ## Adding a new version

  1. Create a new module `BinClass.Model.V2` in `lib/bin_class/model/v2.ex`.
  2. Implement `build/2` in that module.
  3. Add a new clause to `BinClass.Model.build/3`:
     ```elixir
     def build(2, vocab_size, opts), do: BinClass.Model.V2.build(vocab_size, opts)
     ```
  4. Update `@model_version` in `BinClass.Trainer` to `4`.
  """

  @doc """
  Builds the model architecture based on the version.
  """
  def build(version, vocab_size, opts \\ [])

  def build(1, vocab_size, opts) do
    BinClass.Model.V1.build(vocab_size, opts)
  end

  def build(2, vocab_size, opts) do
    BinClass.Model.V2.build(vocab_size, opts)
  end

  def build(3, vocab_size, opts) do
    BinClass.Model.V3.build(vocab_size, opts)
  end

  def build(4, vocab_size, opts) do
    BinClass.Model.V4.build(vocab_size, opts)
  end

  def build(version, _vocab_size, _opts) do
    raise ArgumentError, "Unknown model version: #{version}"
  end
end