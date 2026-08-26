defmodule BinClass.Model do
  @moduledoc """
  Dispatcher for named model architectures.

  ## Adding a new architecture

  1. Create a new module e.g. `BinClass.Model.NewArch` in `lib/bin_class/model/new_arch.ex`.
  2. Implement `build/2` in that module, using the shared internal model
     configuration helpers for option validation and seeded dropout.
  3. Add a new clause to `BinClass.Model.build/3`:
     ```elixir
     def build(:new_arch, vocab_size, opts), do: BinClass.Model.NewArch.build(vocab_size, opts)
     ```
  4. Update the training configuration's default architecture if it should
     become the default.
  """

  alias BinClass.Model.{
    Cnn,
    CnnMixedPooling,
    MultiScaleCnn,
    ParallelCnn,
    SepSeCnn,
    Transformer
  }

  @architectures [
    :cnn,
    :cnn_mixed_pooling,
    :multi_scale_cnn,
    :sep_se_cnn,
    :parallel_cnn,
    :transformer
  ]

  @revision 1

  @minimum_vector_lengths %{
    cnn: 3,
    cnn_mixed_pooling: 3,
    multi_scale_cnn: 5,
    sep_se_cnn: 1,
    parallel_cnn: 1,
    transformer: 1
  }

  @doc """
  Returns the supported architecture names.
  """
  def architectures, do: @architectures

  @doc false
  def revision, do: @revision

  @doc false
  def validate_revision!(@revision), do: @revision

  def validate_revision!(revision) do
    raise ArgumentError,
          "classifier model revision #{inspect(revision)} is incompatible with revision " <>
            "#{@revision}; retrain the classifier with this BinClass version"
  end

  @doc false
  def validate_name!(name) when name in @architectures, do: name

  def validate_name!(name) do
    raise ArgumentError, "Unknown model architecture: #{inspect(name)}"
  end

  @doc false
  def minimum_vector_length(architecture) do
    architecture
    |> validate_name!()
    |> then(&Map.fetch!(@minimum_vector_lengths, &1))
  end

  @doc false
  def validate_vector_length!(architecture, vector_length) do
    minimum = minimum_vector_length(architecture)

    unless is_integer(vector_length) and vector_length > 0 do
      raise ArgumentError,
            ":vector_length must be a positive integer, got: #{inspect(vector_length)}"
    end

    if vector_length < minimum do
      raise ArgumentError,
            ":vector_length must be at least #{minimum} for architecture " <>
              "#{inspect(architecture)}, got: #{vector_length}"
    end

    vector_length
  end

  @doc """
  Builds a named binary-classification model.

  `architecture` must be one of the values returned by `architectures/0`, and
  `vocab_size` must be a positive integer greater than `:pad_token_id`. The
  returned Axon graph accepts batches of token IDs and produces two-class
  probability distributions.

  ## Common options

    * `:embedding_size` - Positive embedding width. Defaults to `64`.
    * `:dropout_rate` - Number from `0.0` up to, but excluding, `1.0`.
      Defaults to `0.2`.
    * `:pad_token_id` - Non-negative padding ID below `vocab_size`. Defaults to
      `0`.
    * `:seed` - Optional integer used to seed dropout layers. Defaults to
      `nil`.

  ## Architecture-specific options

    * `:conv_filters` - Positive filter count for `:cnn` and
      `:cnn_mixed_pooling`. Defaults to `128` and `96`, respectively.
    * `:branch_filters` - Positive per-branch filter count for
      `:multi_scale_cnn` and `:sep_se_cnn`. Defaults to `32` and `48`,
      respectively.
    * `:ff_dim` - Positive feed-forward width for `:transformer`. Defaults to
      `128`.
  """
  def build(architecture, vocab_size, opts \\ [])

  def build(:cnn, vocab_size, opts) do
    Cnn.build(vocab_size, opts)
  end

  def build(:cnn_mixed_pooling, vocab_size, opts) do
    CnnMixedPooling.build(vocab_size, opts)
  end

  def build(:multi_scale_cnn, vocab_size, opts) do
    MultiScaleCnn.build(vocab_size, opts)
  end

  def build(:sep_se_cnn, vocab_size, opts) do
    SepSeCnn.build(vocab_size, opts)
  end

  def build(:parallel_cnn, vocab_size, opts) do
    ParallelCnn.build(vocab_size, opts)
  end

  def build(:transformer, vocab_size, opts) do
    Transformer.build(vocab_size, opts)
  end

  def build(architecture, _vocab_size, _opts) do
    validate_name!(architecture)
  end
end
