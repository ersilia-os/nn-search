import socket, time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pymilvus import (
  connections,
  utility,
  Collection,
  DataType,
)
from pymilvus.exceptions import MilvusException

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.padding import Padding
from rich.progress import Progress

import rich_click as click
import rich_click.rich_click as rc

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True

rc.USE_RICH_MARKUP = True
rc.SHOW_ARGUMENTS = True
rc.COLOR_SYSTEM = "truecolor"
rc.STYLE_OPTION = "bold magenta"
rc.STYLE_COMMAND = "bold green"
rc.STYLE_METAVAR = "italic yellow"
rc.STYLE_SWITCH = "underline cyan"
rc.STYLE_USAGE = "bold blue"
rc.STYLE_OPTION_DEFAULT = "dim italic"

console = Console()


def parse_host_port(uri: str) -> Tuple[str, int]:
  uri = uri.strip()
  if "://" not in uri:
    if ":" in uri:
      host, port = uri.rsplit(":", 1)
      return host, int(port)
    return uri, 19530

  parsed = urlparse(uri)
  host = parsed.hostname or "localhost"
  port = parsed.port or 19530
  return host, int(port)


def tcp_ping(
  host: str, port: int, timeout: float = 1.5
) -> Tuple[bool, Optional[float], Optional[str]]:
  t0 = time.perf_counter()
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.settimeout(timeout)
  try:
    sock.connect((host, port))
    sock.close()
    dt = (time.perf_counter() - t0) * 1000.0
    return True, dt, None
  except Exception as e:
    try:
      sock.close()
    except Exception:
      pass
    return False, None, str(e)


def bytes_human(n: float) -> str:
  units = ["B", "KB", "MB", "GB", "TB", "PB"]
  i = 0
  while n >= 1024 and i < len(units) - 1:
    n /= 1024.0
    i += 1
  return f"{n:.2f} {units[i]}"


def estimate_row_bytes(field_type: DataType, dim: Optional[int]) -> int:
  if field_type == DataType.FLOAT_VECTOR and dim:
    return dim * 4
  if field_type == DataType.BINARY_VECTOR and dim:
    return dim // 8
  return 0


def describe_collection_pretty(name: str) -> Dict:
  info = {
    "name": name,
    "entities": 0,
    "est_size_bytes": 0,
    "fields": [],
    "indexes": [],
    "loaded": False,
  }

  coll = Collection(name)
  try:
    info["entities"] = coll.num_entities
  except MilvusException:
    info["entities"] = 0

  try:
    for f in coll.schema.fields:
      dim = None
      if f.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
        dim = f.params.get("dim") if hasattr(f, "params") else None
        try:
          dim = int(dim) if dim is not None else None
        except Exception:
          dim = None
      info["fields"].append({
        "name": f.name,
        "type": str(f.dtype).replace("DataType.", ""),
        "dim": dim,
      })
  except MilvusException:
    pass

  per_row = 0
  for f in info["fields"]:
    ftype = getattr(DataType, f["type"], None)
    per_row += estimate_row_bytes(ftype, f.get("dim"))
  info["est_size_bytes"] = per_row * info["entities"]

  try:
    idxs = getattr(coll, "indexes", [])
    for idx in idxs:
      meta = getattr(idx, "params", {}) or {}
      info["indexes"].append({
        "field": getattr(idx, "field_name", "?"),
        "index_type": meta.get("index_type", "?"),
        "metric_type": meta.get("metric_type", "?"),
        "params": meta.get("params", {}),
      })
  except Exception:
    try:
      names = utility.list_indexes(collection_name=name)
      for iname in names:
        info["indexes"].append({
          "field": "?",
          "index_type": iname,
          "metric_type": "?",
          "params": {},
        })
    except Exception:
      pass

  try:
    info["loaded"] = coll.is_empty is False and coll.has_index()
  except Exception:
    info["loaded"] = False

  return info


def connect_milvus(
  alias: str, uri: str, token: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
  try:
    try:
      connections.disconnect(alias=alias)
    except Exception:
      pass
    connections.connect(alias=alias, uri=uri, token=token)
    _ = utility.get_server_version()
    return True, None
  except Exception as e:
    return False, str(e)


def common_options(func):
  options = [
    click.option("--input-file", "-i", required=True, help="Path to the input file."),
    click.option(
      "--model-dir",
      "-m",
      required=False,
      default=None,
      help="Directory where the model is stored.",
    ),
    click.option("--anonymize", is_flag=True, help="Whether to anonymize outputs."),
  ]
  for option in reversed(options):
    func = option(func)
  return func


from typing import List, Dict, Optional
from rich.table import Table
from rich import box
from rich.panel import Panel

_TYPE_ABBR = {
  "INT64": "I64",
  "INT32": "I32",
  "INT16": "I16",
  "INT8": "I8",
  "BOOL": "B",
  "DOUBLE": "F64",
  "FLOAT": "F32",
  "VARCHAR": "STR",
  "FLOAT_VECTOR": "FV",
  "BINARY_VECTOR": "BV",
}


def _abbr_type(t: str) -> str:
  return _TYPE_ABBR.get(t, t.replace("DataType.", ""))


def _abbr_index_type(ix: str) -> str:
  return ix.replace("BIN_", "") if isinstance(ix, str) else str(ix)


def _fields_inline(fields: List[Dict], max_items: int = 3) -> str:
  pairs = []
  for f in fields:
    t = _abbr_type(str(f.get("type", "")))
    d = f.get("dim")
    if d:
      pairs.append(f"{f.get('name', '?')}:{t}({d})")
    else:
      pairs.append(f"{f.get('name', '?')}:{t}")
    if len(pairs) >= max_items:
      break
  tail = " …" if len(fields) > max_items else ""
  return ", ".join(pairs) + tail if pairs else "-"


def _indexes_inline(indexes: List[Dict], max_items: int = 2) -> str:
  pairs = []
  for idx in indexes:
    pairs.append(
      f"{idx.get('field', '?')}:{_abbr_index_type(idx.get('index_type', '?'))}"
    )
    if len(pairs) >= max_items:
      break
  tail = " …" if len(indexes) > max_items else ""
  return ", ".join(pairs) + tail if pairs else "-"


def build_ultracompact_collections_table(
  infos: List[Dict],
  *,
  show_indexes: bool = True,
  width: int = 200,
  title: Optional[str] = "Collections",
  panel: bool = True,
):
  tbl = Table(
    title=title,
    box=box.MINIMAL,
    expand=False,
    show_header=True,
    show_lines=False,
    pad_edge=False,
    padding=(0, 0),
  )

  tbl.add_column(
    "Name",
    style="bold",
    no_wrap=True,
    max_width=500,
    overflow="ellipsis",
  )
  tbl.add_column(
    "Ent.", justify="right", no_wrap=True, max_width=8, overflow="ellipsis"
  )
  tbl.add_column("Sz", justify="right", no_wrap=True, max_width=8, overflow="ellipsis")
  fields_max = max(10, int(width * (0.42 if show_indexes else 0.6)))
  tbl.add_column("Fields", no_wrap=False, max_width=30, overflow="ellipsis")
  if show_indexes:
    idx_max = max(8, width - (tbl.columns[0].max_width + 8 + 8 + fields_max + 6))
    tbl.add_column("Idx", no_wrap=False, max_width=200, overflow="ellipsis")

  for info in infos:
    name = str(info.get("name", "?"))
    ents = f"{int(info.get('entities', 0)):,}"
    size = bytes_human(float(info.get("est_size_bytes", 0)))

    ftxt = _fields_inline(info.get("fields", []), max_items=3)
    if show_indexes:
      itxt = _indexes_inline(info.get("indexes", []), max_items=2)
      tbl.add_row(name, ents, size, ftxt, itxt)
    else:
      tbl.add_row(name, ents, size, ftxt)

  if panel:
    return Panel(tbl, border_style="magenta", padding=0, width=tbl.width + 2)
  return tbl


def summerize_status(milvus_uri, alias, token, timeout, show_indexes, *, width=60):
  _TYPE_ABBR = {
    "INT64": "I64",
    "INT32": "I32",
    "INT16": "I16",
    "INT8": "I8",
    "BOOL": "B",
    "DOUBLE": "F64",
    "FLOAT": "F32",
    "VARCHAR": "STR",
    "FLOAT_VECTOR": "FV",
    "BINARY_VECTOR": "BV",
  }

  def _abbr_type(t: str) -> str:
    return _TYPE_ABBR.get(t, t.replace("DataType.", ""))

  def _abbr_index_type(ix: str) -> str:
    return ix.replace("BIN_", "") if isinstance(ix, str) else str(ix)

  def _fields_inline(fields: List[Dict], max_items: int = 3) -> str:
    pairs = []
    for f in fields:
      t = _abbr_type(str(f.get("type", "")))
      d = f.get("dim")
      pairs.append(
        f"{f.get('name', '?')}:{t}({d})" if d else f"{f.get('name', '?')}:{t}"
      )
      if len(pairs) >= max_items:
        break
    tail = " …" if len(fields) > max_items else ""
    return (", ".join(pairs) + tail) if pairs else "-"

  def _indexes_inline(indexes: List[Dict], max_items: int = 2) -> str:
    pairs = []
    for idx in indexes:
      pairs.append(
        f"{idx.get('field', '?')}:{_abbr_index_type(idx.get('index_type', '?'))}"
      )
      if len(pairs) >= max_items:
        break
    tail = " …" if len(indexes) > max_items else ""
    return (", ".join(pairs) + tail) if pairs else "-"

  def _compact_engine_panel(*, title="Engine", width=60, data=None):
    tbl = Table(
      title=title,
      box=box.MINIMAL,
      expand=False,
      show_header=False,
      show_lines=False,
      pad_edge=False,
      padding=(0, 0),
    )
    tbl.width = max(40, width)
    tbl.add_column(
      "K", style="bold cyan", no_wrap=True, max_width=10, overflow="ellipsis"
    )
    tbl.add_column("V", no_wrap=False, overflow="ellipsis")
    for k, v in data or []:
      tbl.add_row(k, v)
    return Panel(tbl, border_style="blue", padding=0, width=tbl.width + 2)

  def _compact_collections_panel(
    infos: List[Dict], *, title="Collections", width=60, show_indexes=True
  ):
    tbl = Table(
      title=title,
      box=box.MINIMAL,
      expand=False,
      show_header=True,
      show_lines=False,
      pad_edge=False,
      padding=(0, 0),
    )
    tbl.width = 80
    name_w = max(8, int(width * 0.28))
    tbl.add_column(
      "Name", style="bold", no_wrap=True, max_width=name_w, overflow="ellipsis"
    )
    tbl.add_column(
      "Entry", justify="right", no_wrap=True, max_width=8, overflow="ellipsis"
    )
    tbl.add_column(
      "Sz", justify="right", no_wrap=True, max_width=8, overflow="ellipsis"
    )
    fields_w = max(10, int(width * (0.42 if show_indexes else 0.6)))
    tbl.add_column("Flds", no_wrap=False, max_width=fields_w, overflow="ellipsis")
    if show_indexes:
      idx_w = max(8, width - (name_w + 8 + 8 + fields_w + 6))
      tbl.add_column("Idx", no_wrap=False, max_width=idx_w, overflow="ellipsis")

    for info in infos:
      name = str(info.get("name", "?"))
      ents = f"{int(info.get('entities', 0)):,}"
      size = bytes_human(float(info.get("est_size_bytes", 0)))
      ftxt = _fields_inline(info.get("fields", []), max_items=3)
      if show_indexes:
        itxt = _indexes_inline(info.get("indexes", []), max_items=2)
        tbl.add_row(name, ents, size, ftxt, itxt)
      else:
        tbl.add_row(name, ents, size, ftxt)

    return Panel(tbl, border_style="magenta", padding=0, width=tbl.width + 2)

  host, port = parse_host_port(milvus_uri)
  ok_tcp, latency_ms, tcp_err = tcp_ping(host, port, timeout=timeout)

  engine_rows = [
    ("URI", f"[bold]{milvus_uri}[/bold]"),
    ("Host", host),
    ("Port", str(port)),
    ("Reach", "[bold green]Yes[/bold green]" if ok_tcp else "[bold red]No[/bold red]"),
    ("TCP(ms)", f"{latency_ms:.1f}" if ok_tcp else f"[red]{tcp_err}[/red]"),
  ]

  connected, conn_err = connect_milvus(alias=alias, uri=milvus_uri, token=token)
  if connected:
    try:
      version = utility.get_server_version()
    except Exception:
      version = "unknown"
    try:
      mode = utility.get_server_mode()
    except Exception:
      mode = "unknown"
    engine_rows += [
      ("SDK", "[bold green]Yes[/bold green]"),
      ("Ver", version),
      ("Mode", mode),
    ]
  else:
    engine_rows += [
      ("SDK", "[bold red]No[/bold red]"),
      ("Err", f"[red]{conn_err}[/red]"),
    ]

  #   console.print(_compact_engine_panel(title="Engine", width=width, data=engine_rows))

  if not connected:
    console.print(
      Padding(
        "[red]Could not query collections without a successful SDK connection.[/red]",
        (1, 0, 0, 0),
      )
    )
    return

  coll_names: List[str] = []
  try:
    coll_names = utility.list_collections()
  except MilvusException as e:
    console.print(
      Panel(
        Text(f"Failed to list collections: {e}", style="red"),
        border_style="red",
        padding=0,
      )
    )
    return

  if not coll_names:
    console.print(Padding("[yellow]No collections found.[/yellow]", (1, 0, 0, 0)))
    return

  infos = []
  with Progress(transient=True) as progress:
    task = progress.add_task("[cyan]Fetching metadata...", total=len(coll_names))
    for name in coll_names:
      info = describe_collection_pretty(name)
      infos.append(info)
      progress.update(task, advance=1)

  console.print(
    _compact_collections_panel(
      infos, title="Collections", width=width, show_indexes=show_indexes
    )
  )


@click.group()
def cli():
  pass


@cli.command()
@common_options
def filter(input_file, model_dir, anonymize):
  pass


@cli.command()
@common_options
@click.option(
  "--output-dir", "-o", required=False, help="Path to the output model dir."
)
@click.option(
  "--override-dir",
  required=False,
  is_flag=True,
  default=False,
  help="Path to the output model dir.",
)
def build(input_file, model_dir, anonymize, output_dir, override_dir):
  pass


if __name__ == "__main__":
  cli()
