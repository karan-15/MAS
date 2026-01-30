import json
from pathlib import Path

import numpy as np
import typer
import yaml

app = typer.Typer(help="Build dense FAISS index from id_to_embedding.npz")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.command()
def main(config: str):
    cfg = load_config(config)

    emb_path = Path(cfg["dense"]["embeddings_path"])
    faiss_path = Path(cfg["dense"]["faiss_path"])
    idmap_path = Path(cfg["dense"]["idmap_path"])

    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    idmap_path.parent.mkdir(parents=True, exist_ok=True)

    # Import faiss here so the script fails early if not installed
    import faiss

    typer.echo(f"[dense_build] Loading embeddings from {emb_path} ...")
    data = np.load(emb_path, allow_pickle=True)

    # We support two common formats:
    # 1) arrays: "ids" and "embeddings"
    # 2) dict-like: keys are doc_ids and values are vectors (less common)
    if "ids" in data.files and "embeddings" in data.files:
        ids = data["ids"]
        X = data["embeddings"]
        ids = [str(x) for x in ids.tolist()]
    else:
        # fallback: treat each key as an id -> vector
        ids = []
        vecs = []
        for k in data.files:
            ids.append(str(k))
            vecs.append(data[k])
        X = np.vstack(vecs)

    # Ensure float32 for FAISS
    X = np.asarray(X, dtype="float32")

    # Normalize if using cosine similarity via inner product
    faiss.normalize_L2(X)

    d = X.shape[1]
    typer.echo(f"[dense_build] embeddings: {X.shape[0]} x {d}")

    # Cosine similarity = inner product after L2 normalization
    index = faiss.IndexFlatIP(d)
    index.add(X)

    faiss.write_index(index, str(faiss_path))
    typer.echo(f"[dense_build] Wrote FAISS index to {faiss_path}")

    idmap_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"[dense_build] Wrote id map to {idmap_path}")


if __name__ == "__main__":
    app()
