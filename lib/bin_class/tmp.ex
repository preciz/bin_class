defmodule BinClass.Tmp do
  @moduledoc false

  @doc """
  Executes a function within a temporary directory.
  The directory is created before the function is called and deleted afterwards.
  """
  def with_tmp_dir(func) do
    base = System.tmp_dir!()
    random = :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
    tmp_path = Path.join(base, "bin_class_#{random}")

    File.mkdir_p!(tmp_path)

    try do
      func.(tmp_path)
    after
      File.rm_rf!(tmp_path)
    end
  end
end
