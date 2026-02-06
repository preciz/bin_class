defmodule BinClassTest do
  use ExUnit.Case
  doctest BinClass

  test "modules are loaded" do
    assert Code.ensure_loaded?(BinClass.Trainer)
    assert Code.ensure_loaded?(BinClass.Serving)
    assert Code.ensure_loaded?(BinClass.Model)
    assert Code.ensure_loaded?(BinClass.Tokenizer)
  end

  test "serialization round-trip with version check" do
    tokenizer = BinClass.Tokenizer.train(["hello world"])
    
    classifier = %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: Nx.tensor([0.1, 0.2], type: :f32) |> Nx.backend_transfer(Nx.BinaryBackend),
      accuracy: 0.9,
      epoch: 1,
      vector_length: 10,
      vocab_size: 100,
      labels: [0, 1],
      model_version: 99 # Invalid version
    }

    serialized = BinClass.serialize(classifier)

    # deserialization should fail because version 99 is unknown
    assert_raise ArgumentError, "Unknown model version: 99", fn ->
      BinClass.deserialize(serialized)
    end
  end

  test "deserialization defaults to version 1" do
    tokenizer = BinClass.Tokenizer.train(["hello world"])
    
    data = %{
      tokenizer_json: BinClass.Tokenizer.to_json(tokenizer),
      model_params: Nx.tensor([0.1, 0.2], type: :f32) |> Nx.backend_transfer(Nx.BinaryBackend),
      vector_length: 10,
      vocab_size: 100,
      labels: [0, 1]
      # Missing model_version
    }

    binary = :erlang.term_to_binary(data)
    
    # Should not raise, assuming version 1 is valid
    serving = BinClass.deserialize(binary)
    assert %Nx.Serving{} = serving
  end
end
