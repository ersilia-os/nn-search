# Approximate Nearest neighbour search
Approximate Nearest neighbour search for the Ersilia Model Hub models result

This project provides a **vector database storage and retrieval tool for SMILES compounds**. Each model ersilia models have its own collection in **Milvus**, storing:

* **Canonical SMILES**
* **Embedding vector**
* **Associated results** 

### How it works

1. **Ingestion**

   * Canonicalize SMILES → compute embeddings → insert into Milvus with results.
   * The best part about this is that we can use existing cache pipeline to fetch bulk model precalculation
2. **Retrieval**

   * Query with a SMILES → embed → search top-K similar compounds in the chosen model’s collection.
   * Returns SMILES, similarity scores, and their stored results.
3. **(Optional)** Re-rank results with chemical similarity (e.g., Tanimoto on fingerprints).

### API Endpoints

* `POST /ingest` → Insert SMILES + embedding + result.
* `POST /query` → Retrieve top-K results for a query SMILES.
* `GET /models` → List available models/collections.

### Tech stack

* **Milvus** for vector storage & search
* **FastAPI** for ingestion/query API
* **RDKit** for canonicalization and optional re-ranking

### How to use it


# CLI

A command-line interface for managing vector databases, filtering data, and controlling the engine.

## Installation
First install uv, a lightening speed project manager as below:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv pip install -e .
```

## Usage

Run the CLI with:

```bash
python -m nns.cli <command> [options]
```

Or if installed:

```bash
nns <command> [options]
```

---

## Commands

### 1. `filter`

Convert CSV to ANN JSON.

```bash
nns filter -i input.csv -o output.json -c eos3b5e -k 10
```

**Options**

* `-i, --input-file` (required) → Input CSV file
* `-o, --output-file` (required) → Output JSON file
* `-c, --collection-name` (required) → Target collection name (model id of ersilia)
* `-k, --topk` → Number of top results (default: 5)

---

### 2. `engine`

Manage the vector DB engine.

```bash
nns engine --start     # Start engine
nns engine --restart   # Restart engine
nns engine --stop      # Stop engine
nns engine --upgrade   # Upgrade engine
nns engine --delete    # Delete engine
```

**Flags**

* `-s, --start`
* `-r, --restart`
* `-p, --stop`
* `-u, --upgrade`
* `-d, --delete`

---

### 3. `status`

Check connection and index details.

```bash
nns status --milvus-uri http://localhost:19530 --alias mydb
```

**Options**

* `--milvus-uri` → Milvus URI (default: `http://localhost:19530`)
* `--alias` → Connection alias (default: `default`)
* `--token` → Auth token (e.g., `root:Milvus`)
* `--timeout` → TCP timeout (default: 1.5s)
* `--show-indexes/--no-show-indexes` (default: show)

---

### 4. `build`

Build vector DB from cloud cache.

```bash
nns build -c eos5axz/eos3b5e
```

**Options**

* `-c, --collection-name` (required)

---

## SImple example workflow

```bash
# 1. Start the milvus engine. If the image is not in the system, it will be pulled.
nns engine --start

# 2. Build database
nns build -c eos3b5e

# 3. Filter data / Note that the data csv file is simply a list of smiles
nns filter -i data.csv -o data.json -c eos3b5e -k 10

# 4. Check status of the collection
nns status
```
