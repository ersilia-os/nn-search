import csv, gzip, io, os, requests, time, warnings
from io import StringIO
from nns.utils.vars import GITHUB_CONTENT_URL, PREDEFINED_COLUMN_FILE
from nns.utils.logging import logger
from nns.utils.progress import download_progress
from typing import Optional
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from pymilvus import (
  connections,
  utility,
  FieldSchema,
  CollectionSchema,
  DataType,
  Collection,
)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def merge_csvs_stdlib(files, output):
  with open(output, "w", newline="") as fout:
    writer = None
    for path in files:
      with open(path, "r", newline="") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        if writer is None:
          writer = csv.writer(fout)
          writer.writerow(header)
        for row in reader:
          writer.writerow(row)


class ApiClient:
  def __init__(
    self,
    base_url: str = None,
    timeout: int = 3600,
    milvus_uri: Optional[str] = None,
    alias: str = "default",
    token: Optional[str] = None,
  ):
    self.base_url = base_url or ""
    self.timeout = timeout
    self.milvus_uri = milvus_uri or os.getenv("MILVUS_URI", "http://localhost:19530")
    self.alias = alias
    self.token = token

  def get_json(self, endpoint: str, params: dict = None):
    r = requests.get(endpoint, params=params, timeout=self.timeout)
    r.raise_for_status()
    return r.json()

  def post_json(self, endpoint: str, json: dict = None):
    r = requests.post(endpoint, json=json, timeout=self.timeout)
    r.raise_for_status()
    return r.json()

  def head(self, url: str):
    r = requests.head(url, timeout=self.timeout)
    r.raise_for_status()
    return r.headers

  def download_stream(self, url: str, chunk_size: int = 64 * 1024):
    r = requests.get(url, stream=True, timeout=self.timeout)
    r.raise_for_status()
    for chunk in r.iter_content(chunk_size):
      yield chunk

  @staticmethod
  def morgan_bytes(smiles: str, bits: int = 1024, radius: int = 2) -> Optional[bytes]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
      return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    arr = np.zeros((bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return np.packbits(arr, bitorder="big").tobytes()

  def ensure_collection(self, name: str, fp_bits: int) -> Collection:
    try:
      connections.disconnect(alias=self.alias)
    except Exception:
      pass
    connections.connect(alias=self.alias, uri=self.milvus_uri, token=self.token)
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

  def ensure_index_loaded(self, coll: Collection, metric: str = "JACCARD"):
    try:
      has = any(
        getattr(ix, "field_name", "") == "fp" for ix in getattr(coll, "indexes", [])
      )
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

  def merge_shards(
    self,
    shards,
    smiles_list,
    downloader,
    model_id,
    *,
    fp_bits: int = 1024,
    batch_size: int = 5000,
  ):
    gz_shards = [
      s
      for s in shards
      if s.get("key", "").endswith(".gz") or s.get("url", "").endswith(".gz")
    ]
    total_size = 0
    for s in gz_shards:
      size = s.get("size", 0)
      try:
        hdrs = downloader.head(s["url"])
        size = int(hdrs.get("Content-Length", size))
      except Exception:
        pass
      total_size += int(size or 0)
    logger.info(f"⬇ Total download size: {total_size / 1024:.1f} KB")
    coll = self.ensure_collection(model_id, fp_bits)
    if not smiles_list:
      logger.info("⬇ No reorder list provided; writing to Milvus")
      smiles_batch, value_batch, fp_batch = [], [], []
      with download_progress(
        desc="⬇ downloading shards", total_bytes=total_size, transient=True
      ) as (progress, task):
        for shard in gz_shards:
          buf = io.BytesIO()
          for chunk in downloader.download_stream(shard["url"]):
            if not chunk:
              continue
            buf.write(chunk)
            progress.update(task, advance=len(chunk))
          text = gzip.decompress(buf.getvalue()).decode("utf-8", errors="replace")
          reader = csv.reader(io.StringIO(text))
          for row in reader:
            # print(row[2:])
            if not row:
              continue
            smi = row[1].strip()
            try:
              val = [float(r) for r in row[2:]]
            except Exception:
              continue
            fp = self.morgan_bytes(smi, bits=fp_bits)
            if fp is None:
              continue
            smiles_batch.append(smi)
            value_batch.append(val)
            fp_batch.append(fp)
            if len(smiles_batch) >= batch_size:
              coll.insert([smiles_batch, value_batch, fp_batch])
              smiles_batch.clear()
              value_batch.clear()
              fp_batch.clear()
      if smiles_batch:
        coll.insert([smiles_batch, value_batch, fp_batch])
      self.ensure_index_loaded(coll, metric="JACCARD")
      logger.info(f"Written to Milvus collection '{model_id}'")
      return


def fetch_schema_from_github(model_id):
  st = time.perf_counter()
  try:
    response = requests.get(
      f"{GITHUB_CONTENT_URL}/{model_id}/main/{PREDEFINED_COLUMN_FILE}"
    )
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


class JobStatus:
  PENDING = "PENDING"
  RUNNING = "RUNNING"
  FAILED = "FAILED"
  SUCCEEDED = "SUCCEEDED"
