import csv, gzip, io, requests
from nns.utils.logging import logger
from nns.utils.helpers import ensure_collection, ensure_index_loaded, morgan_bytes
from nns.utils.progress import download_progress


class ApiClient:
  def __init__(
    self,
    base_url: str = None,
    timeout: int = 3600,
  ):
    self.base_url = base_url or ""
    self.timeout = timeout

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

  def merge_shards(
    self,
    shards,
    downloader,
    model_id,
    *,
    fp_bits: int = 1024,
    batch_size: int = 5000,
  ):
    gz_shards = [
      s for s in shards if s.get("key", "").endswith(".gz") or s.get("url", "").endswith(".gz")
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
    coll = ensure_collection(model_id, fp_bits)
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
          if not row:
            continue
          smi = row[1].strip()
          try:
            val = [float(r) for r in row[2:]]
          except Exception:
            continue
          fp = morgan_bytes(smi, bits=fp_bits)
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
      ensure_index_loaded(coll, metric="JACCARD")
      logger.info(f"Written to Milvus collection '{model_id}'")
      return


class JobStatus:
  PENDING = "PENDING"
  RUNNING = "RUNNING"
  FAILED = "FAILED"
  SUCCEEDED = "SUCCEEDED"
