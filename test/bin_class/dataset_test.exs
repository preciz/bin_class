defmodule BinClass.DatasetTest do
  use ExUnit.Case
  alias BinClass.Dataset

  @sample_data [
    %{text: "great movie loved it a lot", label: 1},
    %{text: "fantastic film absolutely wonderful", label: 1},
    %{text: "terrible movie hated it completely", label: 0},
    %{text: "awful experience very bad", label: 0},
    %{text: "mediocre but acceptable", label: 1},
    %{text: "horrible waste of time", label: 0}
  ]

  describe "summary/2" do
    test "computes accurate summary statistics on binary dataset" do
      summary = Dataset.summary(@sample_data)

      assert summary.total == 6
      assert summary.class_counts == %{"0" => 3, "1" => 3}
      assert summary.class_distribution == %{"0" => 0.5, "1" => 0.5}
      assert summary.balance_ratio == 1.0
      assert summary.character_length.min > 0
      assert summary.character_length.max >= summary.character_length.min
      assert summary.character_length.mean > 0
      assert summary.character_length.median > 0
      assert summary.word_count.min > 0
      assert summary.labels == ["0", "1"]

      formatted = Dataset.format_summary(summary)
      assert formatted =~ "Dataset Summary"
      assert formatted =~ "Total Examples:       6"
      assert formatted =~ "Balance Ratio:        1.0"
    end

    test "supports custom label maps and soft targets" do
      data = [
        {"great text", 0.9},
        {"bad text", 0.2}
      ]

      summary = Dataset.summary(data, labels: %{0 => "negative", 1 => "positive"})
      assert summary.total == 2
      assert summary.class_counts == %{"negative" => 1, "positive" => 1}
      assert summary.labels == ["negative", "positive"]
    end

    test "raises on empty dataset" do
      assert_raise ArgumentError, "dataset must not be empty", fn ->
        Dataset.summary([])
      end
    end

    test "raises on invalid item structure" do
      assert_raise ArgumentError, ~r/Invalid dataset item/, fn ->
        Dataset.summary(["raw string"])
      end
    end
  end

  describe "split/2" do
    test "performs stratified 2-way split" do
      data =
        for id <- 1..100 do
          %{id: id, text: "sample text #{id}", label: if(id <= 30, do: 1, else: 0)}
        end

      {train, val} = Dataset.split(data, validation_split: 0.2, seed: 42)

      assert length(train) == 80
      assert length(val) == 20

      # 30% positive in both splits
      train_pos = Enum.count(train, &(&1.label == 1))
      val_pos = Enum.count(val, &(&1.label == 1))

      assert train_pos == 24
      assert val_pos == 6

      train_ids = MapSet.new(train, & &1.id)
      val_ids = MapSet.new(val, & &1.id)
      assert MapSet.disjoint?(train_ids, val_ids)
    end

    test "performs stratified 3-way train/val/test split" do
      data =
        for id <- 1..100 do
          %{id: id, text: "sample text #{id}", label: rem(id, 2)}
        end

      {train, val, test} = Dataset.split(data, validation_split: 0.1, test_split: 0.1, seed: 42)

      assert length(train) == 80
      assert length(val) == 10
      assert length(test) == 10

      all_ids = Enum.map(train ++ val ++ test, & &1.id)
      assert length(Enum.uniq(all_ids)) == 100
    end

    test "performs unstratified random split when stratify: false" do
      data =
        for id <- 1..50 do
          %{id: id, text: "sample text #{id}", label: rem(id, 2)}
        end

      {train, val} = Dataset.split(data, validation_split: 0.2, stratify: false, seed: 123)
      assert length(train) == 40
      assert length(val) == 10
    end

    test "seed guarantees deterministic reproducible splits" do
      data =
        for id <- 1..50 do
          %{id: id, text: "sample text #{id}", label: rem(id, 2)}
        end

      split1 = Dataset.split(data, validation_split: 0.2, seed: 1234)
      split2 = Dataset.split(data, validation_split: 0.2, seed: 1234)
      split3 = Dataset.split(data, validation_split: 0.2, seed: 5678)

      assert split1 == split2
      refute split1 == split3
    end

    test "validates split ratio boundaries" do
      assert_raise ArgumentError, ~r/:validation_split must be between 0.0 and 1.0/, fn ->
        Dataset.split(@sample_data, validation_split: 1.5)
      end

      assert_raise ArgumentError,
                   ~r/:test_split must be positive and :validation_split \+ :test_split must be < 1.0/,
                   fn ->
                     Dataset.split(@sample_data, validation_split: 0.6, test_split: 0.5)
                   end

      assert_raise ArgumentError, "cannot split an empty dataset", fn ->
        Dataset.split([])
      end
    end
  end
end
