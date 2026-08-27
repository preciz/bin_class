defmodule BinClass.Training.SchedulesTest do
  use ExUnit.Case
  alias BinClass.Trainer
  alias BinClass.Training.Schedules

  test "constant schedule returns constant learning rate" do
    schedule = Schedules.build(:constant, 1.0e-3, 100)
    assert_in_delta Nx.to_number(schedule.(0)), 1.0e-3, 1.0e-6
    assert_in_delta Nx.to_number(schedule.(50)), 1.0e-3, 1.0e-6
    assert_in_delta Nx.to_number(schedule.(100)), 1.0e-3, 1.0e-6
  end

  test "linear decay schedule warms up and decays linearly" do
    schedule = Schedules.build(:linear, 1.0e-3, 100)
    # warmup is 5 steps (0.05 * 100)
    assert_in_delta Nx.to_number(schedule.(0)), 0.0, 1.0e-6
    assert_in_delta Nx.to_number(schedule.(5)), 1.0e-3, 1.0e-6
    # Halfway between step 5 and 100 (approx step 52.5) -> ~0.5e-3
    assert_in_delta Nx.to_number(schedule.(52)), 0.5e-3, 1.0e-4
    # End step
    assert_in_delta Nx.to_number(schedule.(100)), 0.0, 1.0e-6
    assert_in_delta Nx.to_number(schedule.(150)), 0.0, 1.0e-6
  end

  test "linear decay with custom warmup_ratio and end_learning_rate" do
    schedule =
      Schedules.build({:linear, warmup_ratio: 0.1, end_learning_rate: 1.0e-4}, 1.0e-3, 100)

    # warmup is 10 steps (0.1 * 100)
    assert_in_delta Nx.to_number(schedule.(0)), 1.0e-4, 1.0e-6
    assert_in_delta Nx.to_number(schedule.(10)), 1.0e-3, 1.0e-6
    # End step
    assert_in_delta Nx.to_number(schedule.(100)), 1.0e-4, 1.0e-6
  end

  test "cosine decay schedule warms up and decays via cosine curve" do
    schedule = Schedules.build(:cosine, 1.0e-3, 100)
    assert_in_delta Nx.to_number(schedule.(0)), 0.0, 1.0e-6
    assert_in_delta Nx.to_number(schedule.(5)), 1.0e-3, 1.0e-6
    # Cosine at midpoint
    assert_in_delta Nx.to_number(schedule.(52)), 0.5e-3, 1.0e-4
    assert_in_delta Nx.to_number(schedule.(100)), 0.0, 1.0e-6
  end

  test "wsd schedule warms up, stays stable, and decays at the end" do
    schedule = Schedules.build(:wsd, 1.0e-3, 100)
    # Warmup 5% (step 0..5), Stable 75% (step 5..80), Decay 20% (step 80..100)
    assert_in_delta Nx.to_number(schedule.(0)), 0.0, 1.0e-6
    assert_in_delta Nx.to_number(schedule.(5)), 1.0e-3, 1.0e-6
    assert_in_delta Nx.to_number(schedule.(50)), 1.0e-3, 1.0e-6
    assert_in_delta Nx.to_number(schedule.(80)), 1.0e-3, 1.0e-6
    assert_in_delta Nx.to_number(schedule.(90)), 0.5e-3, 1.0e-4
    assert_in_delta Nx.to_number(schedule.(100)), 0.0, 1.0e-6
  end

  test "trains models with different schedules" do
    data = [
      %{text: "first positive example", label: 1},
      %{text: "second positive example", label: 1},
      %{text: "third positive example", label: 1},
      %{text: "first negative example", label: 0},
      %{text: "second negative example", label: 0},
      %{text: "third negative example", label: 0}
    ]

    for schedule <- [:linear, :cosine, :wsd, :constant, {:linear, warmup_ratio: 0.1}] do
      classifier =
        Trainer.train(data,
          epochs: 1,
          batch_size: 2,
          vector_length: 8,
          schedule: schedule
        )

      assert %BinClass.Classifier{} = classifier
      assert classifier.accuracy >= 0
    end
  end

  test "rejects invalid schedule configurations" do
    data = [%{text: "a", label: 1}, %{text: "b", label: 0}]

    for invalid <- [:invalid_schedule, 123, {:unknown_schedule, []}] do
      assert_raise ArgumentError, ~r/:schedule must be/, fn ->
        Trainer.train(data, batch_size: 1, schedule: invalid)
      end
    end

    assert_raise ArgumentError, ~r/:warmup_ratio must be a number between 0.0 and 1.0/, fn ->
      Trainer.train(data, batch_size: 1, schedule: {:linear, warmup_ratio: 1.5})
    end

    assert_raise ArgumentError, ~r/:end_learning_rate must be a non-negative number/, fn ->
      Trainer.train(data,
        batch_size: 1,
        learning_rate: 1.0e-3,
        schedule: {:linear, end_learning_rate: 1.0e-2}
      )
    end
  end
end
