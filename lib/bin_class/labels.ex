defmodule BinClass.Labels do
  @moduledoc false

  @default ["0", "1"]

  def default, do: @default

  def validate!([negative, positive] = labels)
      when is_binary(negative) and is_binary(positive) and negative != positive,
      do: labels

  def validate!(%{0 => negative, 1 => positive} = labels)
      when map_size(labels) == 2 and is_binary(negative) and is_binary(positive) and
             negative != positive,
      do: labels

  def validate!(labels) do
    raise ArgumentError,
          ":labels must be a two-element list of distinct binaries or a map " <>
            "%{0 => negative_binary, 1 => positive_binary}, got: #{inspect(labels)}"
  end
end
