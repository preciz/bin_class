defmodule BinClass.TrainerTest do
  use ExUnit.Case
  alias BinClass.Trainer
  alias BinClass.Training.{Config, Dataset}

  test "splits before balancing and never copies training examples into validation" do
    data = [
      %{id: :positive_1, text: "positive", label: 1},
      %{id: :positive_2, text: "positive", label: 1},
      %{id: :negative_1, text: "negative", label: 0},
      %{id: :negative_2, text: "negative", label: 0},
      %{id: :negative_3, text: "negative", label: 0},
      %{id: :negative_4, text: "negative", label: 0},
      %{id: :negative_5, text: "negative", label: 0},
      %{id: :negative_6, text: "negative", label: 0},
      %{id: :negative_7, text: "negative", label: 0},
      %{id: :negative_8, text: "negative", label: 0}
    ]

    {training, validation} =
      split_and_balance(data,
        batch_size: 1,
        validation_split: 0.2,
        balance: true,
        seed: 1234
      )

    assert length(training) == 14
    assert Enum.count(training, &(&1.label == 1)) == 7
    assert Enum.sort(Enum.map(validation, & &1.label)) == [0, 1]

    training_ids = MapSet.new(training, & &1.id)
    validation_ids = MapSet.new(validation, & &1.id)
    assert MapSet.disjoint?(training_ids, validation_ids)
  end

  test "stratifies imbalanced hard and soft targets when both classes can cover both partitions" do
    hard_data =
      for id <- 1..20 do
        %{id: id, text: "row #{id}", label: if(id <= 2, do: 1, else: 0)}
      end

    {_training, validation} =
      split_and_balance(hard_data,
        batch_size: 2,
        validation_split: 0.1,
        balance: false,
        seed: 1234
      )

    assert Enum.sort(Enum.map(validation, & &1.label)) == [0, 1]

    soft_data = Enum.map(hard_data, &%{&1 | label: if(&1.label == 1, do: 0.8, else: 0.2)})

    {_training, validation} =
      split_and_balance(soft_data,
        batch_size: 2,
        validation_split: 0.1,
        balance: false,
        target_mode: :soft,
        seed: 1234
      )

    assert Enum.sort(Enum.map(validation, & &1.label)) == [0.2, 0.8]
  end

  test "balance_data/1 covers all branches" do
    # Branch c1 > c0
    data_c1_gt_c0 = [
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "neg", label: 0},
      %{text: "neg", label: 0}
    ]

    result = Trainer.train(data_c1_gt_c0, epochs: 1, batch_size: 1, vector_length: 8)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0

    # Branch c0 > c1
    data_c0_gt_c1 = [
      %{text: "neg", label: 0},
      %{text: "neg", label: 0},
      %{text: "pos", label: 1}
    ]

    result = Trainer.train(data_c0_gt_c1, epochs: 1, batch_size: 1, vector_length: 8)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0

    # Branch true (equal or one is zero)
    data_equal = [
      %{text: "neg", label: 0},
      %{text: "pos", label: 1}
    ]

    result = Trainer.train(data_equal, epochs: 1, batch_size: 1, vector_length: 8)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0

    # Branch true (one is zero)
    data_one_zero = [
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1},
      %{text: "pos", label: 1}
    ]

    result = Trainer.train(data_one_zero, epochs: 1, batch_size: 1, vector_length: 8)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0
  end

  test "default vector_length is 512" do
    data = [
      %{text: "a", label: 1},
      %{text: "b", label: 0}
    ]

    result = Trainer.train(data, epochs: 1, batch_size: 1)
    assert %BinClass.Classifier{} = result
    assert result.vector_length == 512
    assert result.architecture == :cnn
    assert result.decision_policy == %{strategy: :argmax}
    assert result.minimum_input_tokens == 0
    assert result.labels == ["0", "1"]
  end

  test "decision behavior does not depend on model architecture" do
    data = [
      %{text: "a", label: 1},
      %{text: "b", label: 0}
    ]

    result =
      Trainer.train(data, epochs: 1, batch_size: 1, architecture: :cnn, vector_length: 8)

    assert %BinClass.Classifier{} = result
    assert result.decision_policy == %{strategy: :argmax}
    assert result.minimum_input_tokens == 0
  end

  test "trainer with custom options" do
    data = [
      %{text: "a", label: 1},
      %{text: "b", label: 0},
      %{text: "c", label: 1},
      %{text: "d", label: 0},
      %{text: "e", label: 1},
      %{text: "f", label: 0},
      %{text: "g", label: 1},
      %{text: "h", label: 0},
      %{text: "i", label: 1},
      %{text: "j", label: 0}
    ]

    result =
      Trainer.train(data,
        epochs: 1,
        batch_size: 2,
        learning_rate: 0.01,
        dropout_rate: 0.35,
        decay: 0.1,
        validation_split: 0.2,
        patience: 2,
        decision_policy: {:positive_threshold, 0.7},
        minimum_input_tokens: 1,
        vector_length: 8
      )

    assert %BinClass.Classifier{} = result
    assert result.learning_rate == 0.01
    assert result.dropout_rate == 0.35
    assert result.decision_policy == %{strategy: :positive_threshold, positive_threshold: 0.7}
    assert result.minimum_input_tokens == 1
  end

  test "explicitly optimizes and persists a positive threshold" do
    data =
      for id <- 1..20 do
        %{id: id, text: "row #{id} evidence", label: rem(id, 2)}
      end

    result =
      Trainer.train(data,
        epochs: 1,
        batch_size: 2,
        vector_length: 8,
        architecture: :cnn,
        seed: 1234,
        decision_policy:
          {:optimize_threshold, [objective: :balanced_accuracy, false_positive_cost: 0.5]}
      )

    assert %{
             strategy: :positive_threshold,
             positive_threshold: threshold,
             optimization: %{
               objective: :balanced_accuracy,
               false_positive_cost: 0.5,
               validation_score: _
             }
           } = result.decision_policy

    assert threshold >= 0.0 and threshold <= 1.0
  end

  test "checkpoint selection evaluates and selects the best checkpoint using the configured objective" do
    data =
      for id <- 1..30 do
        %{id: id, text: "sample text #{id} classification feature", label: rem(id, 2)}
      end

    result =
      Trainer.train(data,
        epochs: 2,
        batch_size: 2,
        vector_length: 8,
        architecture: :cnn,
        seed: 42,
        decision_policy: {:optimize_threshold, [objective: :f1]}
      )

    assert %{
             strategy: :positive_threshold,
             optimization: %{
               objective: :f1,
               validation_score: validation_score,
               validation_f1: validation_f1
             }
           } = result.decision_policy

    assert validation_score == validation_f1
    assert result.epoch in [0, 1]
  end

  test "threshold optimization fails before model setup when both partitions cannot contain both classes" do
    insufficient_class =
      for id <- 1..20 do
        %{id: id, text: "row #{id}", label: if(id == 1, do: 1, else: 0)}
      end

    assert_raise ArgumentError,
                 ~r/threshold optimization needs at least two examples from each class.*19 negative and 1 positive/s,
                 fn ->
                   Trainer.train(insufficient_class,
                     batch_size: 2,
                     vector_length: 8,
                     decision_policy: :optimize_threshold
                   )
                 end

    partition_too_small = [
      %{text: "negative 1", label: 0},
      %{text: "negative 2", label: 0},
      %{text: "positive 1", label: 1},
      %{text: "positive 2", label: 1}
    ]

    assert_raise ArgumentError,
                 ~r/partitions that can each contain both classes.*3 training and 1 validation/s,
                 fn ->
                   Trainer.train(partition_too_small,
                     batch_size: 1,
                     validation_split: 0.25,
                     vector_length: 8,
                     decision_policy: :optimize_threshold
                   )
                 end
  end

  test "explicit vector_length" do
    data = [%{text: "a", label: 1}, %{text: "b", label: 0}]
    result = Trainer.train(data, epochs: 1, batch_size: 1, vector_length: 123)
    assert %BinClass.Classifier{} = result
    assert result.vector_length == 123
  end

  test "custom compiler" do
    data = [%{text: "a", label: 1}, %{text: "b", label: 0}]
    # We use EXLA explicitly
    result = Trainer.train(data, epochs: 1, batch_size: 1, compiler: EXLA, vector_length: 8)
    assert %BinClass.Classifier{} = result
    assert result.accuracy >= 0
  end

  test "rejects invalid hard and soft target configuration before training" do
    assert_raise ArgumentError, ~r/invalid hard target 0.2/, fn ->
      Trainer.train([%{text: "invalid", label: 0.2}], target_mode: :hard)
    end

    assert_raise ArgumentError, ~r/invalid soft target 1.1/, fn ->
      Trainer.train([%{text: "invalid", label: 1.1}], target_mode: :soft)
    end

    assert_raise ArgumentError, ~r/:target_mode must be :hard or :soft/, fn ->
      Trainer.train([], target_mode: :ordinal)
    end

    assert_raise ArgumentError, ~r/:target_threshold must be a number between 0 and 1/, fn ->
      Trainer.train([], target_mode: :soft, target_threshold: 1.1)
    end

    assert_raise ArgumentError, ~r/:seed must be an integer or nil/, fn ->
      Trainer.train([], seed: :random)
    end

    assert_raise ArgumentError, ~r/:calibrate_threshold/, fn ->
      Trainer.train([], calibrate_threshold: true)
    end
  end

  test "rejects invalid training options before model setup" do
    data = [%{text: "negative", label: 0}, %{text: "positive", label: 1}]

    for {option, value, message} <- [
          {:balance, :yes, ~r/:balance must be a boolean/},
          {:learning_rate, 0, ~r/:learning_rate must be a positive number/},
          {:learning_rate, :fast, ~r/:learning_rate must be a positive number/},
          {:decay, -0.1, ~r/:decay must be a non-negative number/},
          {:patience, -1, ~r/:patience must be a non-negative integer/},
          {:patience, 1.5, ~r/:patience must be a non-negative integer/},
          {:dropout_rate, -0.1, ~r/:dropout_rate must be a number/},
          {:dropout_rate, 1, ~r/:dropout_rate must be a number/},
          {:compiler, String, ~r/:compiler must be a module implementing Nx.Defn.Compiler/}
        ] do
      assert_raise ArgumentError, message, fn ->
        Trainer.train(data, Keyword.put([batch_size: 1], option, value))
      end
    end

    explosive_stream = Stream.map([:unused], fn _ -> raise "training data was consumed" end)

    assert_raise ArgumentError, ~r/:learning_rate must be a positive number/, fn ->
      Trainer.train(explosive_stream, learning_rate: 0)
    end

    assert_raise ArgumentError, ":epochs must be a positive integer, got: 0", fn ->
      Trainer.train(data, epochs: 0, batch_size: 1)
    end

    assert_raise ArgumentError, ":batch_size must be a positive integer, got: 0", fn ->
      Trainer.train(data, batch_size: 0)
    end

    for validation_split <- [0, 1, -0.1, 1.1, :invalid] do
      assert_raise ArgumentError, ~r/:validation_split must be a number/, fn ->
        Trainer.train(data, batch_size: 1, validation_split: validation_split)
      end
    end

    assert_raise ArgumentError, ":vector_length must be a positive integer, got: 0", fn ->
      Trainer.train(data, batch_size: 1, vector_length: 0)
    end

    assert_raise ArgumentError, ~r/must be at least 3 for architecture :cnn/, fn ->
      Trainer.train(data, batch_size: 1, architecture: :cnn, vector_length: 2)
    end

    assert_raise ArgumentError, ~r/must be at least 5 for architecture :multi_scale_cnn/, fn ->
      Trainer.train(data, batch_size: 1, architecture: :multi_scale_cnn, vector_length: 4)
    end

    for labels <- [[:negative, :positive], ["same", "same"], ["only one"], %{0 => "negative"}] do
      assert_raise ArgumentError, ~r/:labels must be.*distinct binaries/, fn ->
        Trainer.train(data, batch_size: 1, labels: labels)
      end
    end

    assert_raise ArgumentError, ~r/:decision_policy must be/, fn ->
      Trainer.train(data, batch_size: 1, decision_policy: :automatic)
    end

    assert_raise ArgumentError, ~r/positive threshold must be a number between 0 and 1/, fn ->
      Trainer.train(data, batch_size: 1, decision_policy: {:positive_threshold, 1.1})
    end

    assert_raise ArgumentError, ~r/:minimum_input_tokens must be a non-negative integer/, fn ->
      Trainer.train(data, batch_size: 1, minimum_input_tokens: -1)
    end

    assert_raise ArgumentError, ~r/:minimum_input_tokens cannot exceed :vector_length/, fn ->
      Trainer.train(data, batch_size: 1, vector_length: 8, minimum_input_tokens: 9)
    end
  end

  test "rejects empty data and configurations with empty batch partitions" do
    assert_raise ArgumentError, "training data must not be empty", fn ->
      Trainer.train([])
    end

    assert_raise ArgumentError, "training data must not be empty", fn ->
      Trainer.train([], batch_size: 1)
    end

    data =
      for id <- 1..10 do
        %{text: "row #{id}", label: rem(id, 2)}
      end

    assert_raise ArgumentError,
                 ~r/must produce at least one full training batch and one full validation batch.*10 examples.*batch_size 6.*0 training batches.*1 validation batches/s,
                 fn ->
                   Trainer.train(data, batch_size: 6, validation_split: 0.2)
                 end
  end

  test "rejects minimum input validation that excludes every validation example" do
    data = [
      %{text: "negative", label: 0},
      %{text: "negative", label: 0},
      %{text: "positive", label: 1},
      %{text: "positive", label: 1}
    ]

    assert_raise ArgumentError, ~r/excludes every validation example/, fn ->
      Trainer.train(data,
        epochs: 1,
        batch_size: 1,
        validation_split: 0.5,
        vector_length: 8,
        minimum_input_tokens: 8,
        seed: 1234
      )
    end
  end

  test "seed makes partitioning and balancing deterministic" do
    data =
      for id <- 1..20 do
        %{id: id, text: "row #{id}", label: rem(id, 2)}
      end

    config =
      Config.new!(
        batch_size: 2,
        validation_split: 0.2,
        balance: true,
        vector_length: 8,
        seed: 1234
      )

    assert Dataset.split_and_balance(data, config) == Dataset.split_and_balance(data, config)

    refute Dataset.split_and_balance(data, config) ==
             Dataset.split_and_balance(data, %{config | seed: 5678})
  end

  defp split_and_balance(data, opts) do
    config = opts |> Keyword.put_new(:vector_length, 8) |> Config.new!()
    Dataset.split_and_balance(data, config)
  end
end
