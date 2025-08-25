import hashlib, json, csv

from redis import ConnectionError, Redis

from nns.utils.vars import REDIS_HOST, REDIS_PORT, REDIS_EXPIRATION
from nns.utils.helpers import (
  ensure_collection,
  ensure_index_loaded,
  morgan_bytes,
  resolve_dtype,
  fetch_schema_from_github,
)

BATCH_SIZE = 5000


class DumpLocalCache:
  def __init__(self, file_name: str, model_id: str):
    self.file_name = file_name
    self.model_id = model_id
    self.schema = fetch_schema_from_github(self.model_id)
    assert self.schema is not None, "Model schema can not be fetched from github."
    self.dtype = resolve_dtype(self.schema[1])

  def conn_redis(self):
    return Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

  def init_redis(self):
    global redis_client
    try:
      redis_client = self.conn_redis()
      return True
    except ConnectionError:
      redis_client = None
      return False

  def dump_from_file(self):
    fpb, sb, vb = [], [], []
    collection = ensure_collection(name=self.model_id, fp_bits=1024)

    with open(self.file_name, "r", newline="") as f:
      reader = csv.reader(f)
      next(reader)
      for r in reader:
        fpb.append(morgan_bytes(r[1]))
        sb.append(r[1])
        vb.append([self.dtype(r) for r in r[2:]])
        if len(sb) >= BATCH_SIZE:
          collection.insert([sb, vb, fpb])
          sb.clear
          fpb.clear
          vb.clear
      if sb:
        collection.insert([sb, vb, fpb])
        ensure_index_loaded(collection, metric="JACCARD")

  def generate_redis_key(self, raw_string):
    return hashlib.md5(raw_string.encode()).hexdigest()

  def fetch_cached_results(self, model_id, data, dim):
    hash_key = f"cache:{model_id}"
    fields = [
      item["input"] if isinstance(item, dict) and "input" in item else item for item in data
    ]
    raw = redis_client.hmget(hash_key, fields)
    results, missing = [], []
    for item, val in zip(data, raw):
      if val:
        results.append(json.loads(val))
      else:
        results.append([""] * dim)
        missing.append(item)
    return results, missing

  def fetch_all_cached(self, model_id, dtype, cols):
    def dict_to_lists(d):
      keys = list(d.keys())
      values = list(d.values())
      return keys, values

    hash_key = f"cache:{model_id}"
    header = self.fetch_or_cache_header(model_id)
    header = header or cols
    assert header is not None, (
      "Headers can not be empty! This might happened either the header is not cached or resolved from model schema."
    )
    raw = redis_client.hgetall(hash_key)
    results = {field: json.loads(val) for field, val in raw.items()}
    inputs, results = dict_to_lists(results)
    return results, inputs

  def fetch_or_cache_header(self, model_id, computed_headers=None):
    header_key = f"{model_id}:header"
    cached = redis_client.get(header_key)
    if cached:
      return json.loads(cached) if isinstance(cached, str) else cached
    if computed_headers is not None:
      redis_client.setex(header_key, REDIS_EXPIRATION, json.dumps(computed_headers))
      return computed_headers
    return None

  def list_cached_inputs(self, model_id):
    hash_key = f"cache:{model_id}"
    return redis_client.hkeys(hash_key)

  def make_hashable(self, obj):
    if isinstance(obj, list):
      return tuple(self.make_hashable(x) for x in obj)
    elif isinstance(obj, dict):
      return tuple(sorted((k, self.make_hashable(v)) for k, v in obj.items()))
    return obj
