defmodule BinClass.TrainerPartitionPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias BinClass.Trainer

  property "seeded stratified partitions and balancing preserve their invariants" do
    check all(partition_case <- partition_case(), max_runs: 250) do
      %{
        data_size: data_size,
        positive_count: positive_count,
        batch_size: batch_size,
        validation_split: validation_split,
        seed: seed
      } = partition_case

      data =
        for id <- 1..data_size do
          %{id: id, text: "row #{id}", label: if(id <= positive_count, do: 1, else: 0)}
        end

      {training, validation} =
        Trainer.split_and_balance_data(
          data,
          batch_size,
          validation_split,
          false,
          :hard,
          0.5,
          seed
        )

      assert {training, validation} ==
               Trainer.split_and_balance_data(
                 data,
                 batch_size,
                 validation_split,
                 false,
                 :hard,
                 0.5,
                 seed
               )

      {expected_training_count, expected_validation_count} =
        partition_counts(data_size, batch_size, validation_split)

      assert length(training) == expected_training_count
      assert length(validation) == expected_validation_count

      input_ids = MapSet.new(data, & &1.id)
      training_ids = MapSet.new(training, & &1.id)
      validation_ids = MapSet.new(validation, & &1.id)

      assert MapSet.subset?(training_ids, input_ids)
      assert MapSet.subset?(validation_ids, input_ids)
      assert MapSet.disjoint?(training_ids, validation_ids)
      assert MapSet.size(training_ids) == length(training)
      assert MapSet.size(validation_ids) == length(validation)

      negative_count = data_size - positive_count

      if positive_count >= 2 and negative_count >= 2 and expected_training_count >= 2 and
           expected_validation_count >= 2 do
        assert labels(training) == MapSet.new([0, 1])
        assert labels(validation) == MapSet.new([0, 1])
      end

      {balanced_training, balanced_validation} =
        Trainer.split_and_balance_data(
          data,
          batch_size,
          validation_split,
          true,
          :hard,
          0.5,
          seed
        )

      assert balanced_validation == validation
      assert MapSet.disjoint?(MapSet.new(balanced_training, & &1.id), validation_ids)

      if labels(training) == MapSet.new([0, 1]) do
        assert Enum.count(balanced_training, &(&1.label == 0)) ==
                 Enum.count(balanced_training, &(&1.label == 1))
      end

      soft_data =
        Enum.map(data, fn row ->
          %{row | label: if(row.label == 0, do: 0.25, else: 0.75)}
        end)

      {soft_training, soft_validation} =
        Trainer.split_and_balance_data(
          soft_data,
          batch_size,
          validation_split,
          false,
          :soft,
          0.5,
          seed
        )

      assert Enum.map(soft_training, & &1.id) == Enum.map(training, & &1.id)
      assert Enum.map(soft_validation, & &1.id) == Enum.map(validation, & &1.id)
    end
  end

  defp partition_case do
    generator =
      gen all(
            data_size <- integer(2..80),
            positive_count <- integer(0..data_size),
            batch_size <- integer(1..8),
            validation_step <- integer(1..9),
            seed <- integer(-1_000_000..1_000_000)
          ) do
        %{
          data_size: data_size,
          positive_count: positive_count,
          batch_size: batch_size,
          validation_split: validation_step / 10,
          seed: seed
        }
      end

    filter(generator, fn config ->
      {training_count, validation_count} =
        partition_counts(config.data_size, config.batch_size, config.validation_split)

      training_count > 0 and validation_count > 0
    end)
  end

  defp partition_counts(data_size, batch_size, validation_split) do
    batch_count = div(data_size, batch_size)
    training_batch_count = floor(batch_count * (1.0 - validation_split))
    validation_batch_count = batch_count - training_batch_count

    {training_batch_count * batch_size, validation_batch_count * batch_size}
  end

  defp labels(rows), do: MapSet.new(rows, & &1.label)
end
