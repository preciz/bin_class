defmodule BinClass.Model.Config do
  @moduledoc false

  alias BinClass.Options

  @common_defaults [embedding_size: 64, dropout_rate: 0.2, pad_token_id: 0, seed: nil]

  @doc false
  def validate_options!(vocab_size, opts, architecture_defaults) do
    Options.validate_positive_integer!(vocab_size, :vocab_size)

    opts = Keyword.validate!(opts, Keyword.merge(@common_defaults, architecture_defaults))

    [:embedding_size | Keyword.keys(architecture_defaults)]
    |> Enum.each(fn option ->
      opts
      |> Keyword.fetch!(option)
      |> Options.validate_positive_integer!(option)
    end)

    dropout_rate =
      opts
      |> Keyword.fetch!(:dropout_rate)
      |> Options.validate_dropout_rate!()

    pad_token_id =
      opts
      |> Keyword.fetch!(:pad_token_id)
      |> Options.validate_non_negative_integer!(:pad_token_id)

    if pad_token_id >= vocab_size do
      raise ArgumentError,
            ":pad_token_id must be below :vocab_size (#{vocab_size}), got: #{pad_token_id}"
    end

    opts
    |> Keyword.fetch!(:seed)
    |> Options.validate_optional_integer!(:seed)

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
