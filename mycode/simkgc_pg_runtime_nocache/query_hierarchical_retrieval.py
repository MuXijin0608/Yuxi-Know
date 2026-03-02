import argparse
import json
from pathlib import Path

# 运行示例：
# python query_hierarchical_retrieval.py \
#   --query "进水阀油压系统故障导致机械停机" \
#   --chroma-dir chroma_db_full \
#   --l1-collection simkgc_power_l1_full \
#   --l0-collection simkgc_power_l0_full \
#   --text-encoder sentence-transformers/all-MiniLM-L6-v2


def build_query_embedding(text: str, encoder_name: str) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(encoder_name)
    emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=False)
    return emb[0].astype("float32").tolist()


def query_hierarchical(
    query_text: str,
    chroma_dir: Path,
    l1_collection_name: str,
    l0_collection_name: str,
    text_encoder: str,
    l1_topk: int,
    l0_topk_per_community: int,
) -> dict:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    col_l1 = client.get_collection(l1_collection_name)
    col_l0 = client.get_collection(l0_collection_name)

    query_emb = build_query_embedding(query_text, text_encoder)

    l1_res = col_l1.query(
        query_embeddings=[query_emb],
        n_results=l1_topk,
        include=["documents", "metadatas", "distances"],
    )

    communities = []
    for i in range(len(l1_res["ids"][0])):
        cid = int(l1_res["metadatas"][0][i]["community_id"])
        communities.append(
            {
                "community_id": cid,
                "l1_id": l1_res["ids"][0][i],
                "distance": float(l1_res["distances"][0][i]),
                "summary": l1_res["documents"][0][i],
            }
        )

    l0_hits = []
    for c in communities:
        cid = c["community_id"]
        l0_res = col_l0.query(
            query_embeddings=[query_emb],
            n_results=l0_topk_per_community,
            where={"community_id": cid},
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for j in range(len(l0_res["ids"][0])):
            metadata = l0_res["metadatas"][0][j] or {}
            hits.append(
                {
                    "entity_id": l0_res["ids"][0][j],
                    "distance": float(l0_res["distances"][0][j]),
                    "entity": metadata.get("entity", ""),
                    "community_id": int(metadata.get("community_id", cid)),
                    "document": l0_res["documents"][0][j],
                }
            )

        l0_hits.append({"community_id": cid, "hits": hits})

    return {
        "query": query_text,
        "l1_topk": l1_topk,
        "l0_topk_per_community": l0_topk_per_community,
        "l1_communities": communities,
        "l0_results": l0_hits,
    }


def pretty_print(result: dict) -> None:
    print("=" * 88)
    print(f"Query: {result['query']}")
    print(f"L1 topk={result['l1_topk']}, L0 topk/community={result['l0_topk_per_community']}")
    print("=" * 88)

    for idx, community in enumerate(result["l1_communities"], start=1):
        cid = community["community_id"]
        print(f"\n[L1-{idx}] community_id={cid}, distance={community['distance']:.6f}")
        print(f"summary: {community['summary']}")

        matched = [x for x in result["l0_results"] if x["community_id"] == cid]
        if not matched:
            print("  (no L0 hits)")
            continue

        for rank, hit in enumerate(matched[0]["hits"], start=1):
            print(
                f"  - L0-{rank}: entity_id={hit['entity_id']}, entity={hit['entity']}, distance={hit['distance']:.6f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query two-stage retrieval: L1 community -> L0 entities")
    parser.add_argument("--query", type=str, required=True, help="Natural-language query text")

    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("results/hier_index/full_run/chroma_db_full"),
        help="Chroma persistent directory",
    )
    parser.add_argument("--l1-collection", type=str, default="simkgc_power_l1_full")
    parser.add_argument("--l0-collection", type=str, default="simkgc_power_l0_full")

    parser.add_argument("--text-encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--l1-topk", type=int, default=3)
    parser.add_argument("--l0-topk-per-community", type=int, default=5)

    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save result json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = query_hierarchical(
        query_text=args.query,
        chroma_dir=args.chroma_dir,
        l1_collection_name=args.l1_collection,
        l0_collection_name=args.l0_collection,
        text_encoder=args.text_encoder,
        l1_topk=args.l1_topk,
        l0_topk_per_community=args.l0_topk_per_community,
    )

    pretty_print(result)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved json -> {args.save_json}")


if __name__ == "__main__":
    main()
