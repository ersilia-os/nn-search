import os, csv, json, numpy as np
from collections.abc import Sequence
from pymilvus import connections, Collection
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from nns.utils.logging import logger


def morgan_bytes(smi: str, bits: int = 1024, radius: int = 2) -> bytes:
  m = Chem.MolFromSmiles(smi)
  if m is None:
    return b""
  bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=bits)
  a = np.zeros((bits,), dtype=np.uint8)
  DataStructs.ConvertToNumpyArray(bv, a)
  return np.packbits(a, bitorder="big").tobytes()


def _values_as_float_list(v):
  if v is None:
    return []
  if isinstance(v, np.ndarray):
    return [float(x) for x in v.tolist()]
  if isinstance(v, (np.floating, np.integer)):
    return [float(v.item())]
  if isinstance(v, (int, float)):
    return [float(v)]
  if isinstance(v, (str, bytes, bytearray)):
    s = v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else v
    s = s.strip()
    try:
      j = json.loads(s)
      if isinstance(j, list):
        return [float(x) for x in j]
      return [float(j)]
    except Exception:
      return []
  if isinstance(v, Sequence):
    if isinstance(v, (str, bytes, bytearray)):
      return []
    return [float(x) for x in list(v)]
  try:
    return [float(x) for x in list(v)]
  except Exception:
    return []


def ann_from_csv_to_json(
  input_csv: str,
  output_path: str,
  collection: str,
  uri: str = None,
  alias: str = "default",
  token: str = None,
  bits: int = 1024,
  topk: int = 5,
):
  uri = uri or os.getenv("MILVUS_URI", "http://localhost:19530")
  try:
    connections.disconnect(alias=alias)
  except:
    pass
  connections.connect(alias=alias, uri=uri, token=token)
  coll = Collection(collection)
  coll.load()

  fields = {f.name for f in coll.schema.fields}
  val_field = "values" if "values" in fields else ("value" if "value" in fields else None)
  out_fields = ["id", "smiles"] + ([val_field] if val_field else [])

  smiles = []
  with open(input_csv, "r", encoding="utf-8") as f:
    r = csv.reader(f)
    try:
      h = next(r)
      if h and h[0].strip().lower() != "smiles":
        s0 = h[0].strip()
        if s0:
          smiles.append(s0)
    except StopIteration:
      pass
    for row in r:
      if not row:
        continue
      s = row[0].strip()
      if s:
        smiles.append(s)

  logger.info(f"Read {len(smiles)} query SMILES from {input_csv}")

  vecs, qmap = [], []
  for i, s in enumerate(smiles):
    b = morgan_bytes(s, bits=bits)
    if not b:
      logger.warning(f"Invalid SMILES skipped: {s}")
      continue
    vecs.append(b)
    qmap.append((i, s))

  logger.info(f"Running ANN search on collection '{collection}' with topk={topk}")

  res = coll.search(
    data=vecs,
    anns_field="fp",
    param={"metric_type": "JACCARD", "params": {"nprobe": 16}},
    limit=topk,
    output_fields=out_fields,
  )

  records = []
  for (qi, qsmi), hits in zip(qmap, res):
    items = []
    for rank, h in enumerate(hits, start=1):
      item = {
        "rank": rank,
        "distance": float(h.distance),
        "id": h.entity.get("id"),
        "smiles": h.entity.get("smiles"),
      }
      if val_field:
        item[val_field] = _values_as_float_list(h.entity.get(val_field))
      items.append(item)
    records.append({"query_smiles": qsmi, "query_index": qi, "results": items})

  if output_path.endswith(".jsonl"):
    with open(output_path, "w", encoding="utf-8") as f:
      for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
  else:
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(records, f, ensure_ascii=False, indent=2)

  logger.info(f"Wrote results to {output_path}")
