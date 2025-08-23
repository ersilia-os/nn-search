from typing import Iterable, TypeVar, Optional
from contextlib import contextmanager
from rich.progress import (
  Progress,
  SpinnerColumn,
  TextColumn,
  BarColumn,
  TaskProgressColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
  DownloadColumn,
  TransferSpeedColumn,
)


T = TypeVar("T")


def make_progress(transient: bool = True) -> Progress:
  return Progress(
    SpinnerColumn(),
    TextColumn("[bold cyan]{task.fields[desc]}[/]"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn(
      "{task.completed}/{task.total}" if "{task.total}" else "{task.completed}"
    ),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    transient=transient,
  )


def track_chunks(
  chunks: Iterable[T], desc: str, total: Optional[int] = None, transient: bool = True
):
  with make_progress(transient) as progress:
    task = progress.add_task("work", total=total, desc=desc)
    for chunk in chunks:
      yield chunk
      try:
        n = len(chunk)
      except Exception:
        n = getattr(chunk, "shape", [1])[0] if hasattr(chunk, "shape") else 1
      progress.update(task, advance=int(n))


def make_download_progress(transient: bool = True) -> Progress:
  return Progress(
    SpinnerColumn(),
    TextColumn("[bold cyan]{task.fields[desc]}[/]"),
    BarColumn(),
    DownloadColumn(binary_units=True),
    TransferSpeedColumn(),
    TimeRemainingColumn(),
    transient=transient,
  )


@contextmanager
def download_progress(
  desc: str, total_bytes: Optional[int] = None, transient: bool = True
):
  with make_download_progress(transient=transient) as progress:
    task_id = progress.add_task("download", total=total_bytes or 0, desc=desc)
    yield progress, task_id
