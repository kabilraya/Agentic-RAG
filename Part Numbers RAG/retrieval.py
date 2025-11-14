from qdrant_client import QdrantClient,models
import re
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

#GLOBAL CONFIG
collection_name = "Misumi Products"
client = QdrantClient(url="http://localhost:6333")
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_encoder = SparseTextEmbedding("Qdrant/bm25")
late_encoder = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

def retrieval(query):
    #embedding the query

    dense_query = next(dense_encoder.query_embed(query))
    sparse_query = next(sparse_encoder.query_embed(query=query))
    late_query = next(late_encoder.query_embed(query=query))

    prefetch = [
        models.Prefetch(
            query=dense_query,
            using="dense",
            limit=20,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_query.as_object()),
            using = "sparse",
            limit = 20
        )
    ]

    points = client.query_points(
        collection_name = collection_name,
        query=late_query,
        using= "lateinteract",
        prefetch=prefetch,
        limit = 10,
        with_payload = True,
        with_vectors=False,

    ).points
    
    return points

def retrieve_part_numbers(query):
    parts = retrieval(query)
    part_numbers = []
    for part in parts:
        part_number = part.payload.get("part_info")
        part_numbers.append(part_number)
    return part_numbers

def main():
    query = "cam followers crown hex socket with width = 7"
    parts = retrieve_part_numbers(query=query)

    for part in parts:
        print(f'{part} \n\n')

if __name__ == "__main__":
    main()