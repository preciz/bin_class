defmodule BinClass.Model do
  @moduledoc """
  Dispatcher for model architectures.

  This module handles model versioning, ensuring that trained models can be
  loaded even if the library's default architecture changes.

  ## Adding a new architecture

  1. Create a new module e.g. `BinClass.Model.NewArch` in `lib/bin_class/model/new_arch.ex`.
  2. Implement `build/2` in that module.
  3. Add a new clause to `BinClass.Model.build/3`:
     ```elixir
     def build(:new_arch, vocab_size, opts), do: BinClass.Model.NewArch.build(vocab_size, opts)
     ```
  4. Update `@model_version` in `BinClass.Trainer` to `:new_arch`.
  """

  @doc """
  Builds the model architecture based on the version.
  """
  def build(version, vocab_size, opts \\ [])

  def build(:cnn, vocab_size, opts) do
    BinClass.Model.Cnn.build(vocab_size, opts)
  end

  def build(:cnn_mixed_pooling, vocab_size, opts) do
    BinClass.Model.CnnMixedPooling.build(vocab_size, opts)
  end

  def build(:multi_scale_cnn, vocab_size, opts) do
    BinClass.Model.MultiScaleCnn.build(vocab_size, opts)
  end

  def build(:sep_se_cnn, vocab_size, opts) do
    BinClass.Model.SepSeCnn.build(vocab_size, opts)
  end

  def build(:parallel_cnn, vocab_size, opts) do
    BinClass.Model.ParallelCnn.build(vocab_size, opts)
  end

  # Backwards compatibility
  def build(1, vocab_size, opts), do: build(:cnn, vocab_size, opts)
  def build(2, vocab_size, opts), do: build(:cnn_mixed_pooling, vocab_size, opts)
  def build(3, vocab_size, opts), do: build(:multi_scale_cnn, vocab_size, opts)
  def build(4, vocab_size, opts), do: build(:sep_se_cnn, vocab_size, opts)
  def build(5, vocab_size, opts), do: build(:parallel_cnn, vocab_size, opts)

  def build(version, _vocab_size, _opts) do
    raise ArgumentError, "Unknown model version: #{inspect(version)}"
  end
end
