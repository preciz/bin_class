defmodule BinClass.Trainer do
  @moduledoc """
  Trains binary text classifiers and returns complete `BinClass.Classifier` values.

  Training includes tokenizer construction, stratified train/validation partitioning,
  optional class balancing, checkpoint selection, and decision-policy calibration.
  """

  alias BinClass.{Model, Tokenizer}
  alias BinClass.Training.{Config, Dataset, Runner, Schedules}

  @doc """
  Trains a binary classifier on the given data stream.

  ## Options

    * `:epochs` - Positive integer number of training epochs. Defaults to `10`.
    * `:batch_size` - Positive integer batch size for training. Defaults to `32`.
    * `:learning_rate` - Positive initial learning rate. Defaults to `1.0e-3`.
    * `:decay` - Non-negative AdamW weight decay. Defaults to `1.0e-2`.
    * `:schedule` - Learning rate decay schedule. Accepts `:linear` (Meta FAIR warmup + linear
      decay), `:cosine` (warmup + cosine decay), `:wsd` (warmup-stable-decay), `:constant`,
      options tuples like `{:linear, warmup_ratio: 0.1}`, or a 1-argument schedule function.
      Defaults to `:linear`.
    * `:labels` - Output label names as a two-element list of distinct binaries or
      `%{0 => negative_binary, 1 => positive_binary}`. Defaults to `["0", "1"]`.
    * `:target_mode` - How training labels are interpreted. `:hard` accepts only `0` and `1`;
      `:soft` accepts positive-class probabilities from `0.0` through `1.0`. Defaults to `:hard`.
    * `:target_threshold` - Boundary used to turn soft targets into binary labels for balancing,
      validation metrics, and threshold calibration. Defaults to `0.5`.
    * `:balance` - Boolean controlling whether to oversample the smaller class in the training
      partition. Validation examples are never oversampled. Soft targets are assigned to a class using
      `:target_threshold`. Defaults to `true`.
    * `:validation_split` - Fraction of full batches to use for validation. Partitions are
      stratified when both target classes can occur in both. Defaults to `0.1`.
    * `:patience` - Non-negative integer number of epochs to wait for improvement before early
      stopping. Defaults to `5`.
    * `:compiler` - A module implementing `Nx.Defn.Compiler`. Defaults to `EXLA`.
    * `:architecture` - The named architecture to use. Defaults to `:parallel_cnn`.
    * `:dropout_rate` - Dropout rate from `0.0` up to, but excluding, `1.0`. Defaults to `0.2`.
    * `:decision_policy` - How probabilities become labels. Accepts `:argmax`,
      `{:positive_threshold, threshold}`, `:optimize_threshold`, or
      `{:optimize_threshold, options}`. Threshold optimization options are `:objective`
      (`:accuracy`, `:balanced_accuracy`, or `:f1`) and `:false_positive_cost`.
      Defaults to `:argmax`.
    * `:minimum_input_tokens` - Inputs with fewer active tokens return
      `:insufficient_input` at inference. Defaults to `0`.
    * `:vector_length` - Positive fixed sequence length satisfying the architecture's minimum.
      Defaults to `512`.
    * `:vocab_size` - Positive maximum vocabulary size for tokenizer training. Defaults to `8_000`.
    * `:normalizer` - Text normalizer to apply before tokenization. Accepts `:nfkc_lowercase`
      (default), `:nfkc`, `:lowercase`, `:none`, `nil`, or a `Tokenizers.Normalizer` struct.
    * `:seed` - Optional integer seed for deterministic partition ordering, model initialization,
      and dropout. Defaults to nondeterministic training.
  """
  def train(data_stream, opts \\ []) do
    config = Config.new!(opts)
    data = Dataset.prepare!(data_stream, config)

    tokenizer =
      Tokenizer.train(
        Stream.map(data, & &1.text),
        vocab_size: config.vocab_size,
        normalizer: config.normalizer
      )

    tokenizer_metadata = Tokenizer.metadata(tokenizer)
    vocab_size = tokenizer_metadata.vocab_size
    pad_token_id = tokenizer_metadata.pad_token_id

    {training_examples, validation_examples} = Dataset.split_and_balance(data, config)

    {training_data, training_labels} =
      Dataset.build_batches(training_examples, tokenizer, config, pad_token_id)

    {validation_data, validation_labels} =
      Dataset.build_batches(validation_examples, tokenizer, config, pad_token_id)

    model_options =
      [dropout_rate: config.dropout_rate, pad_token_id: pad_token_id]
      |> put_seed(config.seed)

    build_model_fn = fn ->
      Model.build(config.architecture, vocab_size, model_options)
    end

    total_steps = config.epochs * length(training_data)
    learning_rate_schedule = Schedules.build(config.schedule, config.learning_rate, total_steps)

    optimizer =
      Polaris.Optimizers.adamw(
        learning_rate: learning_rate_schedule,
        decay: config.decay
      )

    train_model =
      Axon.Loop.trainer(
        build_model_fn.(),
        :categorical_cross_entropy,
        optimizer,
        put_seed([], config.seed)
      )

    training_result =
      %Runner{
        train_model: train_model,
        build_model_fn: build_model_fn,
        training_data: Stream.zip(training_data, training_labels),
        validation_data: Stream.zip(validation_data, validation_labels),
        config: config,
        pad_token_id: pad_token_id
      }
      |> Runner.run()

    %BinClass.Classifier{
      tokenizer: tokenizer,
      model_params: training_result.model_params,
      accuracy: training_result.accuracy,
      epoch: training_result.epoch,
      vector_length: config.vector_length,
      vocab_size: vocab_size,
      pad_token_id: pad_token_id,
      unk_token_id: tokenizer_metadata.unk_token_id,
      labels: config.labels,
      architecture: config.architecture,
      model_revision: Model.revision(),
      learning_rate: config.learning_rate,
      dropout_rate: config.dropout_rate,
      decision_policy: training_result.decision_policy,
      minimum_input_tokens: config.minimum_input_tokens,
      target_mode: config.target_mode,
      target_threshold: config.target_threshold
    }
  end

  defp put_seed(opts, nil), do: opts
  defp put_seed(opts, seed), do: Keyword.put(opts, :seed, seed)
end
