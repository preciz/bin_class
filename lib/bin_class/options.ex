defmodule BinClass.Options do
  @moduledoc false

  def validate_positive_integer!(value, _option) when is_integer(value) and value > 0, do: value

  def validate_positive_integer!(value, option) do
    raise ArgumentError, "#{inspect(option)} must be a positive integer, got: #{inspect(value)}"
  end

  def validate_non_negative_integer!(value, _option)
      when is_integer(value) and value >= 0,
      do: value

  def validate_non_negative_integer!(value, option) do
    raise ArgumentError,
          "#{inspect(option)} must be a non-negative integer, got: #{inspect(value)}"
  end

  def validate_positive_number!(value, _option) when is_number(value) and value > 0, do: value

  def validate_positive_number!(value, option) do
    raise ArgumentError, "#{inspect(option)} must be a positive number, got: #{inspect(value)}"
  end

  def validate_non_negative_number!(value, _option)
      when is_number(value) and value >= 0,
      do: value

  def validate_non_negative_number!(value, option) do
    raise ArgumentError,
          "#{inspect(option)} must be a non-negative number, got: #{inspect(value)}"
  end

  def validate_boolean!(value, _option) when is_boolean(value), do: value

  def validate_boolean!(value, option) do
    raise ArgumentError, "#{inspect(option)} must be a boolean, got: #{inspect(value)}"
  end

  def validate_optional_integer!(nil, _option), do: nil
  def validate_optional_integer!(value, _option) when is_integer(value), do: value

  def validate_optional_integer!(value, option) do
    raise ArgumentError, "#{inspect(option)} must be an integer or nil, got: #{inspect(value)}"
  end

  def validate_dropout_rate!(value)
      when is_number(value) and value >= 0 and value < 1,
      do: value

  def validate_dropout_rate!(value) do
    raise ArgumentError,
          ":dropout_rate must be a number greater than or equal to 0 and below 1, " <>
            "got: #{inspect(value)}"
  end

  def validate_keyword_list!(value, option) when is_list(value) do
    if Keyword.keyword?(value), do: value, else: invalid_keyword_list!(value, option)
  end

  def validate_keyword_list!(value, option), do: invalid_keyword_list!(value, option)

  def validate_compiler!(compiler) when is_atom(compiler) do
    if Code.ensure_loaded?(compiler) and function_exported?(compiler, :__jit__, 5) and
         function_exported?(compiler, :__compile__, 4) do
      compiler
    else
      invalid_compiler!(compiler)
    end
  end

  def validate_compiler!(compiler), do: invalid_compiler!(compiler)

  defp invalid_keyword_list!(value, option) do
    raise ArgumentError, "#{inspect(option)} must be a keyword list, got: #{inspect(value)}"
  end

  defp invalid_compiler!(compiler) do
    raise ArgumentError,
          ":compiler must be a module implementing Nx.Defn.Compiler, got: #{inspect(compiler)}"
  end
end
