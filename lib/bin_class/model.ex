defmodule BinClass.Model do
  @moduledoc """
  Dispatcher for named model architectures.

  ## Adding a new architecture

  1. Create a new module e.g. `BinClass.Model.NewArch` in `lib/bin_class/model/new_arch.ex`.
  2. Implement `build/2` in that module.
  3. Add a new clause to `BinClass.Model.build/3`:
     ```elixir
     def build(:new_arch, vocab_size, opts), do: BinClass.Model.NewArch.build(vocab_size, opts)
     ```
  4. Update `@architecture` in `BinClass.Trainer` to `:new_arch` if it should become the default.
  """

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

  @common_option_defaults [embedding_size: 64, dropout_rate: 0.2, pad_token_id: 0, seed: nil]

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
  Builds a named model architecture.
  """
  def build(architecture, vocab_size, opts \\ [])

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

  def build(:transformer, vocab_size, opts) do
    BinClass.Model.Transformer.build(vocab_size, opts)
  end

  def build(architecture, _vocab_size, _opts) do
    validate_name!(architecture)
  end

  @doc false
  def validate_build_options!(vocab_size, opts, architecture_defaults \\ []) do
    BinClass.Options.validate_positive_integer!(vocab_size, :vocab_size)

    opts = Keyword.validate!(opts, Keyword.merge(@common_option_defaults, architecture_defaults))

    [:embedding_size | Keyword.keys(architecture_defaults)]
    |> Enum.each(fn option ->
      opts
      |> Keyword.fetch!(option)
      |> BinClass.Options.validate_positive_integer!(option)
    end)

    dropout_rate =
      opts
      |> Keyword.fetch!(:dropout_rate)
      |> BinClass.Options.validate_dropout_rate!()

    pad_token_id =
      opts
      |> Keyword.fetch!(:pad_token_id)
      |> BinClass.Options.validate_non_negative_integer!(:pad_token_id)

    if pad_token_id >= vocab_size do
      raise ArgumentError,
            ":pad_token_id must be below :vocab_size (#{vocab_size}), got: #{pad_token_id}"
    end

    opts
    |> Keyword.fetch!(:seed)
    |> BinClass.Options.validate_optional_integer!(:seed)

    Keyword.put(opts, :dropout_rate, dropout_rate)
  end

  @doc false
  def dropout_options(nil, rate, _offset), do: [rate: rate]

  def dropout_options(seed, rate, offset) when is_integer(seed),
    do: [rate: rate, seed: seed + offset]

  def dropout_options(seed, _rate, _offset) do
    raise ArgumentError, ":seed must be an integer or nil, got: #{inspect(seed)}"
  end
end
