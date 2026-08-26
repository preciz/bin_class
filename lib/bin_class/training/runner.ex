defmodule BinClass.Training.Runner do
  @moduledoc false

  alias BinClass.{Evaluation, Padding, Tmp}
  alias BinClass.Training.{Config, Dataset}

  @enforce_keys [
    :train_model,
    :build_model_fn,
    :training_data,
    :validation_data,
    :config,
    :pad_token_id
  ]

  defstruct @enforce_keys

  @doc false
  def run(%__MODULE__{config: %Config{} = config} = runner) do
    loop_options = [epochs: config.epochs, compiler: config.compiler]

    Tmp.with_tmp_dir(fn dir ->
      checkpoints_dir = Path.join(dir, "checkpoints")

      runner.train_model
      |> Axon.Loop.validate(runner.build_model_fn.(), runner.validation_data)
      |> Axon.Loop.early_stop("validation_loss", patience: config.patience)
      |> Axon.Loop.checkpoint(
        event: :epoch_completed,
        path: checkpoints_dir,
        file_pattern: &"#{&1.epoch}.ckpt"
      )
      |> Axon.Loop.run(runner.training_data, Axon.ModelState.empty(), loop_options)

      model = runner.build_model_fn.()
      {_, predict_fn} = Axon.build(model, compiler: config.compiler)

      best_checkpoint =
        checkpoints_dir
        |> Path.join("*.ckpt")
        |> Path.wildcard()
        |> Enum.map(fn checkpoint_path ->
          epoch = checkpoint_path |> Path.basename(".ckpt") |> String.to_integer()
          checkpoint = checkpoint_path |> File.read!() |> Axon.Loop.deserialize_state()

          metrics =
            calculate_validation_metrics(
              predict_fn,
              checkpoint.step_state.model_state,
              runner.validation_data,
              config,
              runner.pad_token_id,
              %{strategy: :argmax},
              0
            )

          %{
            epoch: epoch,
            validation_score: metrics.accuracy,
            checkpoint: checkpoint
          }
        end)
        |> Enum.max_by(& &1.validation_score)

      model_state = best_checkpoint.checkpoint.step_state.model_state

      decision_policy =
        resolve_decision_policy(
          config.decision_config,
          predict_fn,
          model_state,
          runner.validation_data,
          config,
          runner.pad_token_id
        )

      deployment_metrics =
        calculate_validation_metrics(
          predict_fn,
          model_state,
          runner.validation_data,
          config,
          runner.pad_token_id,
          decision_policy,
          config.minimum_input_tokens
        )

      %{
        model_params: model_state,
        accuracy: deployment_metrics.accuracy,
        epoch: best_checkpoint.epoch,
        decision_policy: decision_policy
      }
    end)
  end

  defp calculate_validation_metrics(
         predict_fn,
         trained_model_state,
         validation_data,
         config,
         pad_token_id,
         decision_policy,
         minimum_input_tokens
       ) do
    validation_data
    |> prediction_batches(predict_fn, trained_model_state)
    |> calculate_prediction_metrics(
      config,
      pad_token_id,
      decision_policy,
      minimum_input_tokens
    )
  end

  defp prediction_batches(validation_data, predict_fn, trained_model_state) do
    Stream.map(validation_data, fn {inputs, labels} ->
      {predict_fn.(trained_model_state, inputs), inputs, labels}
    end)
  end

  defp calculate_prediction_metrics(
         prediction_batches,
         config,
         pad_token_id,
         decision_policy,
         minimum_input_tokens
       ) do
    {correct, total, false_positives, true_negatives, insufficient_inputs} =
      Enum.reduce(prediction_batches, {0, 0, 0, 0, 0}, fn {predictions, inputs, labels},
                                                          {correct_acc, count_acc, fp_acc, tn_acc,
                                                           insufficient_acc} ->
        predicted_labels = prediction_labels(predictions, decision_policy)
        true_labels = Dataset.target_labels(labels, config)
        eligible_inputs = eligible_inputs(inputs, pad_token_id, minimum_input_tokens)

        {correct, eligible_count, false_positives, true_negatives} =
          validation_counts(predicted_labels, true_labels, eligible_inputs)

        batch_size = Nx.shape(inputs) |> elem(0)

        {
          correct_acc + correct,
          count_acc + eligible_count,
          fp_acc + false_positives,
          tn_acc + true_negatives,
          insufficient_acc + batch_size - eligible_count
        }
      end)

    if total == 0 do
      raise ArgumentError,
            ":minimum_input_tokens excludes every validation example; lower it or provide " <>
              "longer validation inputs"
    end

    false_positive_denominator = false_positives + true_negatives

    false_positive_rate =
      if false_positive_denominator == 0 do
        0.0
      else
        false_positives / false_positive_denominator
      end

    %{
      accuracy: correct / total,
      false_positive_rate: false_positive_rate,
      false_positives: false_positives,
      true_negatives: true_negatives,
      insufficient_inputs: insufficient_inputs
    }
  end

  defp validation_counts(predicted_labels, true_labels, eligible_inputs) do
    correct =
      predicted_labels
      |> Nx.equal(true_labels)
      |> Nx.logical_and(eligible_inputs)
      |> Nx.sum()

    eligible_count = Nx.sum(eligible_inputs)
    predicted_positive? = Nx.equal(predicted_labels, 1)
    true_negative? = Nx.equal(true_labels, 0)

    false_positives =
      predicted_positive?
      |> Nx.logical_and(true_negative?)
      |> Nx.logical_and(eligible_inputs)
      |> Nx.sum()

    true_negatives =
      predicted_labels
      |> Nx.equal(0)
      |> Nx.logical_and(true_negative?)
      |> Nx.logical_and(eligible_inputs)
      |> Nx.sum()

    [correct, eligible_count, false_positives, true_negatives]
    |> Nx.stack()
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_flat_list()
    |> List.to_tuple()
  end

  defp prediction_labels(predictions, %{strategy: :argmax}) do
    Nx.argmax(predictions, axis: -1)
  end

  defp prediction_labels(predictions, %{
         strategy: :positive_threshold,
         positive_threshold: threshold
       }) do
    batch_size = Nx.shape(predictions) |> elem(0)

    positive? =
      predictions
      |> Nx.slice([0, 1], [batch_size, 1])
      |> Nx.squeeze(axes: [1])
      |> Nx.greater_equal(threshold)

    Nx.select(
      positive?,
      Nx.broadcast(1, Nx.shape(positive?)),
      Nx.broadcast(0, Nx.shape(positive?))
    )
  end

  defp eligible_inputs(inputs, pad_token_id, minimum_input_tokens) do
    inputs
    |> Padding.active_token_counts(pad_token_id)
    |> Nx.greater_equal(minimum_input_tokens)
  end

  defp resolve_decision_policy(
         {:fixed, policy},
         _predict_fn,
         _trained_model_state,
         _validation_data,
         _config,
         _pad_token_id
       ),
       do: policy

  defp resolve_decision_policy(
         {:optimize, optimization},
         predict_fn,
         trained_model_state,
         validation_data,
         config,
         pad_token_id
       ) do
    prediction_batches =
      validation_data
      |> prediction_batches(predict_fn, trained_model_state)
      |> Enum.to_list()

    calibration_options =
      optimization
      |> Map.to_list()
      |> Keyword.put(:eligible, :eligible)

    result =
      prediction_batches
      |> calibration_rows(config, pad_token_id)
      |> Evaluation.calibrate(calibration_options)

    %{
      strategy: :positive_threshold,
      positive_threshold: result.threshold,
      optimization:
        Map.merge(optimization, %{
          validation_score: result.score,
          validation_accuracy: result.accuracy,
          validation_balanced_accuracy: result.balanced_accuracy,
          validation_f1: result.f1,
          validation_false_positive_rate: result.false_positive_rate
        })
    }
  end

  defp calibration_rows(prediction_batches, config, pad_token_id) do
    Enum.flat_map(prediction_batches, fn {predictions, inputs, labels} ->
      batch_size = Nx.shape(inputs) |> elem(0)

      positive_probabilities =
        predictions
        |> Nx.slice([0, 1], [batch_size, 1])
        |> Nx.squeeze(axes: [1])

      eligible =
        inputs
        |> Padding.active_token_counts(pad_token_id)
        |> Nx.greater_equal(config.minimum_input_tokens)

      true_labels = Dataset.target_labels(labels, config)

      [positive_probabilities, Nx.as_type(eligible, :f32), Nx.as_type(true_labels, :f32)]
      |> Nx.stack(axis: 1)
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_list()
      |> Enum.map(fn [probability, eligible?, positive?] ->
        %{
          probability: probability,
          eligible: eligible? == 1.0,
          label: positive? == 1.0
        }
      end)
    end)
  end
end
