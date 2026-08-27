defmodule BinClass.EvaluationTest do
  use ExUnit.Case
  alias BinClass.Evaluation

  @sample_train_data [
    %{text: "excellent brilliant wonderful fantastic good positive", label: 1},
    %{text: "terrible awful horrible bad unpleasant negative", label: 0},
    %{text: "very good and pleasant experience loved it", label: 1},
    %{text: "very bad and disappointing outcome hated it", label: 0}
  ]

  @sample_test_data [
    %{text: "wonderful outcome fantastic", label: 1},
    %{text: "horrible terrible bad", label: 0},
    %{text: "good experience", label: 1},
    %{text: "awful experience", label: 0}
  ]

  describe "evaluate/3 across all architectures" do
    for arch <- BinClass.Model.architectures() do
      @tag :evaluation
      test "evaluates #{arch} model cleanly" do
        arch = unquote(arch)

        classifier =
          BinClass.train(@sample_train_data,
            architecture: arch,
            epochs: 1,
            batch_size: 2,
            vector_length: 16,
            vocab_size: 100
          )

        result = BinClass.evaluate(classifier, @sample_test_data)

        assert is_map(result)
        assert result.total == 4
        assert is_float(result.accuracy)
        assert is_float(result.balanced_accuracy)
        assert is_float(result.precision)
        assert is_float(result.recall)
        assert is_float(result.specificity)
        assert is_float(result.f1_score)
        assert is_float(result.loss)
        assert %{tp: _, fp: _, tn: _, fn: _} = result.confusion_matrix
        assert %{"0" => 2, "1" => 2} = result.class_distribution
        assert result.labels == ["0", "1"]

        # Formatted report
        report = BinClass.format_report(result)
        assert is_binary(report)
        assert report =~ "Evaluation Report"
        assert report =~ "Accuracy:"
        assert report =~ "Confusion Matrix:"
      end
    end
  end

  describe "data format flexibility" do
    setup do
      classifier =
        BinClass.train(@sample_train_data,
          epochs: 1,
          batch_size: 2,
          vector_length: 16,
          vocab_size: 100,
          labels: %{0 => "negative", 1 => "positive"}
        )

      %{classifier: classifier}
    end

    test "supports tuple pairs", %{classifier: classifier} do
      data = [
        {"wonderful fantastic", 1},
        {"horrible awful", 0}
      ]

      result = BinClass.evaluate(classifier, data)
      assert result.total == 2
      assert result.labels == ["negative", "positive"]
    end

    test "supports string key maps", %{classifier: classifier} do
      data = [
        %{"text" => "wonderful fantastic", "label" => "positive"},
        %{"text" => "horrible awful", "label" => "negative"}
      ]

      result = BinClass.evaluate(classifier, data)
      assert result.total == 2
      assert result.labels == ["negative", "positive"]
    end

    test "supports soft float labels and custom threshold", %{classifier: classifier} do
      data = [
        %{text: "wonderful fantastic", label: 0.9},
        %{text: "horrible awful", label: 0.1}
      ]

      result = BinClass.evaluate(classifier, data, target_threshold: 0.5)
      assert result.total == 2
      assert is_float(result.accuracy)
      assert result.labels == ["negative", "positive"]
    end

    test "evaluates anonymous predictor functions directly" do
      predictor = fn
        texts when is_list(texts) ->
          Enum.map(texts, fn _ -> %{label: "1", probabilities: %{"0" => 0.1, "1" => 0.9}} end)

        _text ->
          %{label: "1", probabilities: %{"0" => 0.1, "1" => 0.9}}
      end

      data = [
        %{text: "test a", label: 1},
        %{text: "test b", label: 0}
      ]

      result = Evaluation.evaluate(predictor, data, labels: ["0", "1"])
      assert result.total == 2
      assert result.confusion_matrix.tp == 1
      assert result.confusion_matrix.fp == 1
      assert result.accuracy == 0.5
    end

    test "rejects empty data", %{classifier: classifier} do
      assert_raise ArgumentError, "evaluation data must not be empty", fn ->
        BinClass.evaluate(classifier, [])
      end
    end

    test "rejects invalid item structures", %{classifier: classifier} do
      assert_raise ArgumentError, ~r/Invalid evaluation item/, fn ->
        BinClass.evaluate(classifier, ["just a string"])
      end
    end
  end
end
