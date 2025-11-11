import pandas as pd
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding,TextEmbedding,LateInteractionTextEmbedding
import os
#GLOBAL CONFIG

collection_name = "Misumi Products"
file_name = "camfollower_cam_followers_crown_hex_socket.xlsx"
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_encoder = SparseTextEmbedding("Qdrant/bm25")
late_encoder = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
client = QdrantClient("http://localhost:6333")
main_domain= "https://vn.misumi-ec.com"

def create_collection_with_payloads():
    if not client.collection_exists(collection_name = collection_name):
        client.create_collection(
            collection_name = collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_encoder.embedding_size,
                    distance=models.Distance.COSINE
                ),
                "lateinteract" : models.VectorParams(
                    size=late_encoder.embedding_size,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0)
                )
            },
            sparse_vectors_config={"sparse":models.SparseVectorParams(modifier = models.Modifier.IDF)}
        )
    client.create_payload_index(
        collection_name = collection_name,
        field_name="filename",
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "sub-category",
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="category",
        field_schema = "keyword"
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name = "URL",
        field_schema = "keyword"
    )
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "chunk_id",
        field_schema = "integer"
    )
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "subcategory_number",
        field_schema="integer"
    )

def to_dataframes():
    # loading the excel table into pandas dataframes cleaning the NaN values
    part_numbers_tables = pd.read_excel(file_name)
    part_numbers_tables = part_numbers_tables.fillna("")
    columns_to_join = part_numbers_tables.drop(columns=["Part Number URL"])
    part_numbers_tables["row-text"] = part_numbers_tables.apply(lambda r: " | ".join(f"{c}:{r[c]}" for c in columns_to_join.columns),axis = 1)
    
    

    return part_numbers_tables

def to_vectordb():
    name,ext = os.path.splitext(file_name)
    file_parts = name.split('_')
    category = file_parts[0]
    subcategory = " ".join(file_parts[1:])
    

    create_collection_with_payloads()

    table = to_dataframes()

    part_number_offset = 0
    subcategory_offset = 0

    info = client.get_collection(collection_name = collection_name)
    count = info.points_count
    if(count!=0):
        res,_ = client.scroll(
            collection_name = collection_name,
            limit = 1,
            with_payload=True,
            with_vectors = False,
            order_by={
                "key" : "chunk_id",
                "direction" : "desc"
            }
        )
        if(res):
            part_number = res[0].payload.get("chunk_id")
            subcategory_number = res[0].payload.get("subcategory_number")
            part_number_offset = part_number + 1
            subcategory_offset = subcategory_number + 1
        else:
            part_number_offset = 0
            subcategory_offset = 0
    else:
        part_number_offset = 0
        subcategory_offset = 0
    for idx,row in table.iterrows():
        part_info = f"Category:{category}| Subcategory:{subcategory}" + row['row-text']
        href = row['Part Number URL']
        url = main_domain + href
        client.upsert(
            collection_name = collection_name,
            points=[
                models.PointStruct(
                    id = part_number_offset,
                    payload = {
                        "chunk_id" : part_number_offset,
                        "subcategory_number": subcategory_offset,
                        "URL" : url,
                        "category" : category,
                        "sub-category":subcategory,
                        "part_info" : part_info,
                        "file_name" : file_name,
                        "part_name": row["Part Number Name"]
                    },
                    vector={
                        "dense" : list(dense_encoder.embed(row['row-text']))[0],
                        "sparse" : list(sparse_encoder.embed(row["row-text"]))[0].as_object(),
                        "lateinteract" : list(late_encoder.embed(row["row-text"]))[0]
                    }
                )
            ]
        )
        part_number_offset+=1

def main():
    to_vectordb()

if __name__ == "__main__":
    main()


