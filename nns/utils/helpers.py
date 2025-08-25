import csv, os, requests, socket, time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from nns.utils.vars import GITHUB_CONTENT_URL, PREDEFINED_COLUMN_FILE
from nns.utils.logging import logger

from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus.exceptions import MilvusException

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.padding import Padding
from rich.progress import Progress
from typing import Optional
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from io import StringIO


console = Console()


def fetch_schema_from_github(model_id):
  st = time.perf_counter()
  try:
    response = requests.get(f"{GITHUB_CONTENT_URL}/{model_id}/main/{PREDEFINED_COLUMN_FILE}")
  except requests.RequestException:
    logger.warning("Couldn't fetch column name from github!")
    return None

  csv_data = StringIO(response.text)
  reader = csv.DictReader(csv_data)

  if "name" not in reader.fieldnames:
    logger.warning("Couldn't fetch column name from github. Column name not found.")
    return None

  if "type" not in reader.fieldnames:
    logger.warning("Couldn't fetch data type from github. Column name not found.")
    return None

  rows = list(reader)
  col_name = [row["name"] for row in rows if row["name"]]
  col_dtype = [row["type"] for row in rows if row["type"]]
  shape = len(col_dtype)
  if len(col_name) == 0 and len(col_dtype) == 0:
    return None
  et = time.perf_counter()
  logger.info(f"Column metadata fetched in {et - st:.2} seconds!")
  return col_name, col_dtype, shape


def resolve_dtype(dtype):
  unique_dtypes = list(set(dtype))
  dtype = "float" if "float" in unique_dtypes else unique_dtypes[0]
  if dtype == "integer":
    return int
  if dtype == "float":
    return float
  return str


def morgan_bytes(smiles: str, bits: int = 1024, radius: int = 2) -> Optional[bytes]:
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return None
  bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
  arr = np.zeros((bits,), dtype=np.uint8)
  DataStructs.ConvertToNumpyArray(bv, arr)
  return np.packbits(arr, bitorder="big").tobytes()


def ensure_collection(
  name: str,
  fp_bits: int,
  alias: str = "default",
  milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530"),
  token: Optional[str] = None,
) -> Collection:
  try:
    connections.disconnect(alias=alias)
  except Exception:
    pass
  connections.connect(alias=alias, uri=milvus_uri, token=token)
  if not utility.has_collection(name):
    schema = CollectionSchema([
      FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
      FieldSchema(name="smiles", dtype=DataType.VARCHAR, max_length=512),
      FieldSchema(
        name="values",
        dtype=DataType.ARRAY,
        element_type=DataType.FLOAT,
        max_capacity=2048,
      ),
      FieldSchema(name="fp", dtype=DataType.BINARY_VECTOR, dim=fp_bits),
    ])

    coll = Collection(name=name, schema=schema)
  else:
    coll = Collection(name)
  return coll


def ensure_index_loaded(coll: Collection, metric: str = "JACCARD"):
  try:
    has = any(getattr(ix, "field_name", "") == "fp" for ix in getattr(coll, "indexes", []))
  except Exception:
    has = False
  if not has:
    coll.create_index(
      field_name="fp",
      index_params={
        "index_type": "BIN_IVF_FLAT",
        "metric_type": metric,
        "params": {"nlist": 1024},
      },
    )
  coll.load()


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


def connect_milvus(alias: str, uri: str, token: Optional[str] = None) -> Tuple[bool, Optional[str]]:
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
    pairs.append(f"{idx.get('field', '?')}:{_abbr_index_type(idx.get('index_type', '?'))}")
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
  tbl.add_column("Ent.", justify="right", no_wrap=True, max_width=8, overflow="ellipsis")
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

  SCALAR_BYTE_SIZE = {
    "INT64": 8,
    "INT32": 4,
    "INT16": 2,
    "INT8": 1,
    "DOUBLE": 8,
    "FLOAT": 4,
    "BOOL": 1,
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
      pairs.append(f"{f.get('name', '?')}:{t}({d})" if d else f"{f.get('name', '?')}:{t}")
      if len(pairs) >= max_items:
        break
    tail = " …" if len(fields) > max_items else ""
    return (", ".join(pairs) + tail) if pairs else "-"

  def _indexes_inline(indexes: List[Dict], max_items: int = 2) -> str:
    pairs = []
    for idx in indexes:
      pairs.append(f"{idx.get('field', '?')}:{_abbr_index_type(idx.get('index_type', '?'))}")
      if len(pairs) >= max_items:
        break
    tail = " …" if len(indexes) > max_items else ""
    return (", ".join(pairs) + tail) if pairs else "-"

  def _resolve_bytes(info: Dict) -> float:
    candidate_keys = [
      "est_size_bytes",
      "size_bytes",
      "estimated_bytes",
      "bytes",
      "disk_bytes",
      "storage_bytes",
    ]
    for k in candidate_keys:
      v = info.get(k)
      try:
        if v is not None and float(v) > 0:
          return float(v)
      except Exception:
        pass

    entities = int(info.get("entities", 0)) if info.get("entities") is not None else 0
    fields = info.get("fields", []) or []

    if entities <= 0 or not fields:
      return 0.0

    per_row = 0.0
    for f in fields:
      t = str(f.get("type", "")).replace("DataType.", "")
      if t == "FLOAT_VECTOR":
        dim = int(f.get("dim") or 0)
        per_row += dim * 4
      elif t == "BINARY_VECTOR":
        dim = int(f.get("dim") or 0)
        per_row += max(0, dim // 8)
      elif t == "VARCHAR":
        per_row += 20
      else:
        per_row += SCALAR_BYTE_SIZE.get(t, 0)

    per_row += 8
    est = per_row * entities
    return float(est)

  def _compact_engine_panel(*, title="Engine", width=60, data=None):
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
    tbl.add_column(
      "Key",
      style="bold",
      no_wrap=True,
      max_width=int(width * 0.35),
      overflow="ellipsis",
    )
    tbl.add_column("Value", no_wrap=False, overflow="ellipsis")
    for k, v in data or []:
      tbl.add_row(k, v)
    return Panel(tbl, border_style="magenta", padding=0, width=tbl.width + 2)

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
    tbl.add_column("Name", style="bold", no_wrap=True, max_width=name_w, overflow="ellipsis")
    tbl.add_column("Entry", justify="right", no_wrap=True, max_width=8, overflow="ellipsis")
    tbl.add_column("Sz", justify="right", no_wrap=True, max_width=8, overflow="ellipsis")
    fields_w = max(10, int(width * (0.42 if show_indexes else 0.6)))
    tbl.add_column("Flds", no_wrap=False, max_width=fields_w, overflow="ellipsis")
    if show_indexes:
      idx_w = max(8, width - (name_w + 8 + 8 + fields_w + 6))
      tbl.add_column("Idx", no_wrap=False, max_width=idx_w, overflow="ellipsis")

    for info in infos:
      name = str(info.get("name", "?"))
      ents = f"{int(info.get('entities', 0)):,}"
      size_bytes = _resolve_bytes(info)
      size = bytes_human(size_bytes)
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
      ("Ver", str(version)),
      ("Mode", str(mode)),
    ]
  else:
    engine_rows += [
      ("SDK", "[bold red]No[/bold red]"),
      ("Err", f"[red]{conn_err}[/red]"),
    ]

  console.print(_compact_engine_panel(title="Engine", width=width, data=engine_rows))

  if not connected:
    console.print(
      Padding(
        "[red]Could not query collections without a successful SDK connection.[/red]",
        (1, 0, 0, 0),
      )
    )
    return

  try:
    coll_names: List[str] = utility.list_collections()
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

  infos: List[Dict] = []
  with Progress(transient=True) as progress:
    task = progress.add_task("[cyan]Fetching metadata...", total=len(coll_names))
    for name in coll_names:
      info = describe_collection_pretty(name)
      info.setdefault("name", name)
      info.setdefault("fields", [])
      info.setdefault("entities", info.get("row_count", 0))
      infos.append(info)
      progress.update(task, advance=1)

  console.print(
    _compact_collections_panel(infos, title="Collections", width=width, show_indexes=show_indexes)
  )
