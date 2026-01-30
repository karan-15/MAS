import json
from pathlib import Path

import typer
import yaml
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F

app = typer.Typer(help="Build Whoosh BM25 index from documents.jsonl")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_text(obj: dict) -> tuple[str, str]:
    """
    Returns (title, content) from a document JSON object.
    Supports both the mini corpus and full corpus key variants.
    """
    title = obj.get("title") or ""
    content = (
        obj.get("content")
        or obj.get("post_content")
        or obj.get("description")
        or ""
    )
    return str(title), str(content)


@app.command()
def main(config: str):
    cfg = load_config(config)

    docs_path = Path(cfg["data"]["documents_path"])
    index_dir = Path(cfg["index"]["whoosh_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: field names MUST match retrieval parser fields: ["title", "content"]
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    )

    # Always rebuild cleanly for predictable behavior
    if index.exists_in(index_dir):
        typer.echo(f"[index.build] Index already exists at {index_dir}. Rebuilding...")
        for p in index_dir.glob("*"):
            try:
                p.unlink()
            except IsADirectoryError:
                # In rare cases, Whoosh may create directories
                import shutil
                shutil.rmtree(p, ignore_errors=True)

    ix = index.create_in(index_dir, schema)
    writer = ix.writer(limitmb=512)

    count = 0
    skipped = 0

    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            doc_id = obj.get("id")
            if not doc_id:
                skipped += 1
                continue

            title, content = extract_text(obj)
            if not title and not content:
                skipped += 1
                continue

            writer.add_document(
                doc_id=str(doc_id),
                title=title,
                content=content,
            )
            count += 1

            if count % 50000 == 0:
                typer.echo(f"[index.build] indexed {count} docs...")

    writer.commit()
    typer.echo(f"[index.build] Done. Indexed {count} docs into {index_dir} (skipped={skipped})")

    # ---- Sanity search (VERY useful) ----
    # Query a common term; if this returns 0 consistently, indexing is wrong.
    ix2 = index.open_dir(index_dir)
    with ix2.searcher(weighting=BM25F()) as s:
        qp = QueryParser("content", schema=ix2.schema)
        q = qp.parse("finances")
        r = s.search(q, limit=5)
        typer.echo(f"[index.build] sanity hits for 'the': {len(r)}")
        if len(r) > 0:
            typer.echo(f"[index.build] example doc_id: {r[0]['doc_id']}")

if __name__ == "__main__":
    app()
