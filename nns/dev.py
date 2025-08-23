from pymilvus import connections, utility, MilvusException
from pymilvus import connections, Collection

URI = "http://localhost:19530"


def remove_collection(collection_name):
  connections.connect(alias="default", uri=URI)

  if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Collection '{collection_name}' dropped.")
  else:
    print(f"Collection '{collection_name}' does not exist.")


def get_all_stored(
  name: str, uri="http://localhost:19530", alias="default", token=None, page=1000
):
  try:
    try:
      connections.disconnect(alias=alias)
    except:
      pass
    connections.connect(alias=alias, uri=uri, token=token)
    coll = Collection(name)
    try:
      coll.load()
    except MilvusException:
      coll.load()
    fields = {f.name: f for f in coll.schema.fields}
    id_field = next((f for f in ["id", "pk", "_id"] if f in fields), None)
    val_field = (
      "values" if "values" in fields else ("value" if "value" in fields else None)
    )
    wanted = [x for x in [id_field, "smiles", val_field] if x]
    if not wanted:
      print("No suitable fields found. Available:", list(fields.keys()))
      return []
    offset = 0
    all_rows = []
    while True:
      batch = coll.query(expr="", output_fields=wanted, offset=offset, limit=page)
      if not batch:
        break
      all_rows.extend(batch)
      offset += page
    return all_rows
  except MilvusException as e:
    print("Milvus error:", e)
    return []


if __name__ == "__main__":
  remove_collection("eos5axz")
  remove_collection("eos3b5e")
  #  data = get_all_stored("eos5axz")
  #  print(data)
