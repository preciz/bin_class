defmodule BinClass.DocumentPredictor do
  @moduledoc false

  alias Chunx.Chunker.Token

  def compile(%BinClass.Classifier{} = classifier, opts) do
    opts =
      Keyword.validate!(opts,
        compiler: EXLA,
        batch_size: 16,
        chunk_size: classifier.vector_length,
        chunk_overlap: 0.25
      )

    chunk_size = Keyword.fetch!(opts, :chunk_size)
    chunk_overlap = Keyword.fetch!(opts, :chunk_overlap)

    validate_chunk_size!(chunk_size, classifier.vector_length)
    validate_chunk_overlap!(chunk_overlap, chunk_size)

    predictor =
      BinClass.compile_predictor(classifier,
        compiler: Keyword.fetch!(opts, :compiler),
        batch_size: Keyword.fetch!(opts, :batch_size)
      )

    fn
      text when is_binary(text) ->
        chunks = chunk!(text, classifier.tokenizer, chunk_size, chunk_overlap)
        predictions = predict_chunks(predictor, chunks)

        chunk_results =
          chunks
          |> Enum.zip(predictions)
          |> Enum.with_index()
          |> Enum.map(fn {{chunk, prediction}, index} ->
            %{
              index: index,
              text: chunk.text,
              start_byte: chunk.start_byte,
              end_byte: chunk.end_byte,
              token_count: chunk.token_count,
              prediction: prediction
            }
          end)

        %{
          status: document_status(chunk_results),
          chunks: chunk_results
        }

      input ->
        raise ArgumentError,
              "document predictor expects a single string, got: #{inspect(input)}"
    end
  end

  defp chunk!(text, tokenizer, chunk_size, chunk_overlap) do
    case Token.chunk(text, tokenizer,
           chunk_size: chunk_size,
           chunk_overlap: chunk_overlap
         ) do
      {:ok, chunks} -> chunks
      {:error, reason} -> raise ArgumentError, "could not chunk document: #{inspect(reason)}"
    end
  end

  defp predict_chunks(_predictor, []), do: []
  defp predict_chunks(predictor, chunks), do: predictor.(Enum.map(chunks, & &1.text))

  defp document_status([]), do: :insufficient_input

  defp document_status(chunks) do
    if Enum.any?(chunks, &(&1.prediction.label != :insufficient_input)),
      do: :ok,
      else: :insufficient_input
  end

  defp validate_chunk_size!(chunk_size, vector_length)
       when is_integer(chunk_size) and chunk_size > 0 and chunk_size <= vector_length,
       do: :ok

  defp validate_chunk_size!(chunk_size, vector_length) do
    raise ArgumentError,
          ":chunk_size must be a positive integer no greater than the classifier's " <>
            "vector length (#{vector_length}), got: #{inspect(chunk_size)}"
  end

  defp validate_chunk_overlap!(overlap, chunk_size)
       when is_integer(overlap) and overlap >= 0 and overlap < chunk_size,
       do: :ok

  defp validate_chunk_overlap!(overlap, _chunk_size)
       when is_float(overlap) and overlap >= 0.0 and overlap < 1.0,
       do: :ok

  defp validate_chunk_overlap!(overlap, chunk_size) do
    raise ArgumentError,
          ":chunk_overlap must be a non-negative integer below :chunk_size " <>
            "(#{chunk_size}) or a fraction from 0.0 up to 1.0, got: #{inspect(overlap)}"
  end
end
