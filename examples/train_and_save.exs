# examples/train_and_save.exs

data = [
  %{text: "I loved this movie, it was fantastic!", label: 1},
  %{text: "The acting was great and the plot was engaging.", label: 1},
  %{text: "Absolutely wonderful experience, highly recommend.", label: 1},
  %{text: "Best film I have seen this year!", label: 1},
  %{text: "A masterpiece of modern cinema.", label: 1},
  %{text: "I hated this movie, it was boring.", label: 0},
  %{text: "The plot was weak and the acting was terrible.", label: 0},
  %{text: "A complete waste of time, do not watch.", label: 0},
  %{text: "Worst film ever, I wanted to leave early.", label: 0},
  %{text: "I really did not enjoy this at all.", label: 0}
]

data = List.duplicate(data, 10) |> List.flatten()

# Train
classifier =
  BinClass.Trainer.train(data,
    epochs: 10,
    labels: %{0 => :negative, 1 => :positive},
    validation_split: 0.2
  )

IO.inspect(classifier)

# Save
model_path = "movie_classifier.bin"
BinClass.save(classifier, model_path)
