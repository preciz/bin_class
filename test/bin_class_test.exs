defmodule BinClassTest do
  use ExUnit.Case
  doctest BinClass

  test "modules are loaded" do
    assert Code.ensure_loaded?(BinClass.Trainer)
    assert Code.ensure_loaded?(BinClass.Serving)
    assert Code.ensure_loaded?(BinClass.Model)
    assert Code.ensure_loaded?(BinClass.Tokenizer)
  end

  test "public inference constructors reject unknown options before doing work" do
    classifier = %BinClass.Classifier{}

    for operation <- [
          fn -> BinClass.load("missing.bin", not_an_option: true) end,
          fn -> BinClass.deserialize(<<>>, not_an_option: true) end,
          fn -> BinClass.serving(classifier, not_an_option: true) end,
          fn -> BinClass.compile_predictor(classifier, not_an_option: true) end,
          fn -> BinClass.compile_document_predictor(classifier, not_an_option: true) end
        ] do
      assert_raise ArgumentError, ~r/unknown keys.*:not_an_option/s, operation
    end
  end

  test "serialization rejects numeric model architectures" do
    tokenizer = BinClass.Tokenizer.train(["hello world"])

    classifier = %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: Nx.tensor([0.1, 0.2], type: :f32) |> Nx.backend_transfer(Nx.BinaryBackend),
      accuracy: 0.9,
      epoch: 1,
      vector_length: 10,
      vocab_size: 100,
      labels: ["0", "1"],
      architecture: 99
    }

    assert_raise ArgumentError, "Unknown model architecture: 99", fn ->
      BinClass.serialize(classifier)
    end
  end

  test "deserialization rejects binaries without a named model architecture" do
    tokenizer = BinClass.Tokenizer.train(["hello world"])

    data = %{
      tokenizer_json: BinClass.Tokenizer.to_json(tokenizer),
      model_params: Nx.tensor([0.1, 0.2], type: :f32) |> Nx.backend_transfer(Nx.BinaryBackend),
      vector_length: 10,
      vocab_size: 100,
      labels: ["0", "1"]
      # Missing architecture
    }

    binary = :erlang.term_to_binary(data)

    assert_raise ArgumentError, "Serialized classifier is missing :architecture", fn ->
      BinClass.deserialize_classifier(binary)
    end
  end

  test "deserialization safely rejects terms containing new atoms" do
    atom_name = "bin_class_atom_that_must_not_exist_during_safe_decoding_20260826"
    unsafe_binary = <<131, 100, byte_size(atom_name)::16, atom_name::binary>>

    assert_raise ArgumentError, ~r/invalid or unsafe external representation of a term/, fn ->
      BinClass.deserialize_classifier(unsafe_binary)
    end
  end

  test "serialization and deserialization require binary output labels" do
    tokenizer = BinClass.Tokenizer.train(["hello world"])
    metadata = BinClass.Tokenizer.metadata(tokenizer)

    classifier = %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: Nx.tensor([0.1, 0.2], type: :f32),
      vector_length: 10,
      vocab_size: metadata.vocab_size,
      pad_token_id: metadata.pad_token_id,
      unk_token_id: metadata.unk_token_id,
      labels: ["negative", "positive"],
      architecture: :cnn,
      decision_policy: %{strategy: :argmax},
      minimum_input_tokens: 0,
      target_mode: :hard,
      target_threshold: 0.5
    }

    assert_raise ArgumentError, ~r/:labels must be.*distinct binaries/, fn ->
      BinClass.serialize(%{classifier | labels: [:negative, :positive]})
    end

    binary = BinClass.serialize(classifier)

    tampered_binary =
      binary
      |> :erlang.binary_to_term()
      |> Map.put(:labels, [0, 1])
      |> :erlang.term_to_binary()

    assert_raise ArgumentError, ~r/:labels must be.*distinct binaries/, fn ->
      BinClass.deserialize_classifier(tampered_binary)
    end

    assert %{labels: ["negative", "positive"]} = BinClass.deserialize_classifier(binary)

    BinClass.Tmp.with_tmp_dir(fn dir ->
      path = Path.join(dir, "binary-labels.bin")
      File.write!(path, binary)

      code_path_args =
        :code.get_path()
        |> Enum.flat_map(fn code_path -> ["-pa", List.to_string(code_path)] end)

      expression = """
      Application.ensure_all_started(:bin_class)
      classifier = BinClass.load_classifier(System.fetch_env!("BIN_CLASS_TEST_ARTIFACT"))
      IO.write(Enum.join(classifier.labels, "|"))
      """

      {output, 0} =
        System.cmd(
          System.find_executable("elixir"),
          code_path_args ++ ["-e", expression],
          env: [{"BIN_CLASS_TEST_ARTIFACT", path}]
        )

      assert output == "negative|positive"
    end)
  end

  test "serialization preserves soft target training metadata" do
    tokenizer = BinClass.Tokenizer.train(["hello world"])
    metadata = BinClass.Tokenizer.metadata(tokenizer)

    classifier = %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: Nx.tensor([0.1, 0.2], type: :f32),
      vector_length: 10,
      vocab_size: metadata.vocab_size,
      pad_token_id: metadata.pad_token_id,
      unk_token_id: metadata.unk_token_id,
      labels: ["0", "1"],
      architecture: :cnn,
      decision_policy: %{strategy: :argmax},
      minimum_input_tokens: 0,
      target_mode: :soft,
      target_threshold: 0.6
    }

    binary = BinClass.serialize(classifier)
    serialized_data = :erlang.binary_to_term(binary)
    loaded = BinClass.deserialize_classifier(binary)

    assert serialized_data.architecture == :cnn
    assert serialized_data.model_revision == BinClass.Model.revision()
    assert loaded.architecture == :cnn
    assert loaded.model_revision == BinClass.Model.revision()
    assert loaded.target_mode == :soft
    assert loaded.target_threshold == 0.6
    assert loaded.decision_policy == %{strategy: :argmax}
    assert loaded.minimum_input_tokens == 0
    assert loaded.vocab_size == metadata.vocab_size
    assert loaded.pad_token_id == metadata.pad_token_id
    assert loaded.unk_token_id == metadata.unk_token_id
  end

  test "deserialization rejects missing or incompatible model revisions" do
    tokenizer = BinClass.Tokenizer.train(["hello world"])
    metadata = BinClass.Tokenizer.metadata(tokenizer)

    classifier = %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: Nx.tensor([0.1, 0.2], type: :f32),
      vector_length: 10,
      vocab_size: metadata.vocab_size,
      pad_token_id: metadata.pad_token_id,
      unk_token_id: metadata.unk_token_id,
      labels: ["0", "1"],
      architecture: :cnn,
      decision_policy: %{strategy: :argmax},
      minimum_input_tokens: 0
    }

    serialized = classifier |> BinClass.serialize() |> :erlang.binary_to_term()

    assert_raise ArgumentError, "Serialized classifier is missing :model_revision", fn ->
      serialized
      |> Map.delete(:model_revision)
      |> :erlang.term_to_binary()
      |> BinClass.deserialize_classifier()
    end

    assert_raise ArgumentError, ~r/model revision 0 is incompatible/, fn ->
      serialized
      |> Map.put(:model_revision, 0)
      |> :erlang.term_to_binary()
      |> BinClass.deserialize_classifier()
    end
  end

  test "deserialization rejects tokenizer metadata inconsistent with its JSON" do
    tokenizer = BinClass.Tokenizer.train(["hello world"])
    metadata = BinClass.Tokenizer.metadata(tokenizer)

    classifier = %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: Nx.tensor([0.1, 0.2], type: :f32),
      vector_length: 10,
      vocab_size: metadata.vocab_size,
      pad_token_id: metadata.pad_token_id,
      unk_token_id: metadata.unk_token_id,
      labels: ["0", "1"],
      architecture: :cnn,
      decision_policy: %{strategy: :argmax},
      minimum_input_tokens: 0
    }

    assert_raise ArgumentError, ~r/stored tokenizer metadata vocab_size.*tokenizer JSON/, fn ->
      BinClass.serialize(%{classifier | vocab_size: classifier.vocab_size + 1})
    end

    tampered_binary =
      classifier
      |> BinClass.serialize()
      |> :erlang.binary_to_term()
      |> Map.update!(:pad_token_id, &(&1 + 1))
      |> :erlang.term_to_binary()

    assert_raise ArgumentError, ~r/stored tokenizer metadata pad_token_id.*tokenizer JSON/, fn ->
      BinClass.deserialize_classifier(tampered_binary)
    end
  end

  test "deserialization requires explicit tokenizer metadata" do
    tokenizer = BinClass.Tokenizer.train(["hello world"])
    metadata = BinClass.Tokenizer.metadata(tokenizer)

    data = %{
      tokenizer_json: BinClass.Tokenizer.to_json(tokenizer),
      model_params: Nx.tensor([0.1, 0.2], type: :f32),
      vector_length: 10,
      vocab_size: metadata.vocab_size,
      unk_token_id: metadata.unk_token_id,
      labels: ["0", "1"],
      architecture: :cnn,
      model_revision: BinClass.Model.revision()
    }

    assert_raise ArgumentError, "Serialized classifier is missing :pad_token_id", fn ->
      BinClass.deserialize_classifier(:erlang.term_to_binary(data))
    end
  end

  test "compile_predictor returns working prediction function" do
    data = [
      %{text: "I love this!", label: 1},
      %{text: "I hate this!", label: 0},
      %{text: "I love this!", label: 1},
      %{text: "I hate this!", label: 0},
      %{text: "I love this!", label: 1},
      %{text: "I hate this!", label: 0},
      %{text: "I love this!", label: 1},
      %{text: "I hate this!", label: 0},
      %{text: "I love this!", label: 1},
      %{text: "I hate this!", label: 0},
      %{text: "I love this!", label: 1},
      %{text: "I hate this!", label: 0},
      %{text: "I love this!", label: 1},
      %{text: "I hate this!", label: 0},
      %{text: "I love this!", label: 1},
      %{text: "I hate this!", label: 0},
      %{text: "I love this!", label: 1},
      %{text: "I hate this!", label: 0},
      %{text: "I love this!", label: 1},
      %{text: "I hate this!", label: 0}
    ]

    classifier = BinClass.Trainer.train(data, epochs: 1, batch_size: 2, vector_length: 16)

    predictor = BinClass.compile_predictor(classifier, batch_size: 2)

    serving = BinClass.serving(classifier, batch_size: 2)
    assert %{label: _, confidence: _, probabilities: _} = Nx.Serving.run(serving, "I love this!")

    assert_raise ArgumentError, ~r/architecture/, fn ->
      BinClass.serving(classifier, architecture: :cnn)
    end

    assert_raise ArgumentError, ~r/decision_policy/, fn ->
      BinClass.serving(classifier, decision_policy: %{strategy: :argmax})
    end

    assert_raise ArgumentError, ~r/minimum_input_tokens/, fn ->
      BinClass.serving(classifier, minimum_input_tokens: 1)
    end

    result = predictor.("I love this!")
    assert %{label: _, confidence: _, probabilities: _} = result
    assert result.label in ["0", "1"]

    # Batch input
    results = predictor.(["I love this!", "I hate this!"])
    assert length(results) == 2

    # Inputs larger than the compiled batch size are processed in order without being dropped.
    texts = ["first", "second", "third", "fourth", "fifth"]
    results = predictor.(texts)
    assert length(results) == length(texts)
    assert Enum.all?(results, &match?(%{label: _, confidence: _, probabilities: _}, &1))
    assert results == Enum.map(texts, &predictor.(&1))

    threshold_classifier = %{
      classifier
      | decision_policy: %{strategy: :positive_threshold, positive_threshold: 0.0}
    }

    threshold_classifier =
      threshold_classifier |> BinClass.serialize() |> BinClass.deserialize_classifier()

    threshold_serving = BinClass.serving(threshold_classifier, batch_size: 2)
    threshold_predictor = BinClass.compile_predictor(threshold_classifier, batch_size: 2)

    assert Nx.Serving.run(threshold_serving, "short").label == "1"
    assert threshold_predictor.("short").label == "1"

    padding_classifier = %{threshold_classifier | minimum_input_tokens: 1}
    padding_serving = BinClass.serving(padding_classifier, batch_size: 2)
    padding_predictor = BinClass.compile_predictor(padding_classifier, batch_size: 2)

    assert %{label: :insufficient_input, active_token_count: 0} =
             Nx.Serving.run(padding_serving, "[PAD] [PAD]")

    assert %{label: :insufficient_input, active_token_count: 0} =
             padding_predictor.("[PAD] [PAD]")

    assert Nx.Serving.run(padding_serving, "☃").label == "1"
    assert padding_predictor.("☃").label == "1"

    insufficient_classifier = %{threshold_classifier | minimum_input_tokens: 16}

    insufficient_classifier =
      insufficient_classifier |> BinClass.serialize() |> BinClass.deserialize_classifier()

    insufficient_serving = BinClass.serving(insufficient_classifier, batch_size: 2)
    insufficient_predictor = BinClass.compile_predictor(insufficient_classifier, batch_size: 2)

    serving_result = Nx.Serving.run(insufficient_serving, "short")
    predictor_result = insufficient_predictor.("short")

    assert serving_result.label == :insufficient_input
    assert serving_result.confidence == nil
    assert predictor_result.label == serving_result.label
    assert predictor_result.confidence == serving_result.confidence
    assert Map.keys(predictor_result.probabilities) == Map.keys(serving_result.probabilities)

    Enum.each(serving_result.probabilities, fn {label, probability} ->
      assert_in_delta predictor_result.probabilities[label], probability, 1.0e-6
    end)

    assert serving_result.probabilities != %{}
  end
end
