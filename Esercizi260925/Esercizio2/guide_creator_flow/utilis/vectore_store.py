from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    PayloadSchemaType,
    FieldCondition,
    MatchValue,
    MatchText,
    Filter,
    SearchParams,
    PointStruct,
)


class VectoreStore:

    def __init__(self, client, settings, vector_size, embeddings):
        self.client, self.settings = client, settings
        self.vector_size = vector_size
        self.embeddings = embeddings
        self.recreate_collection_for_rag()

    def recreate_collection_for_rag(self):

        self.client.recreate_collection(
            collection_name=self.settings.collection,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(
                m=32,             # grado medio del grafo HNSW (maggiore = più memoria/qualità)
                ef_construct=256  # ampiezza lista candidati in fase costruzione (qualità/tempo build)
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2  # parallelismo/segmentazione iniziale
            ),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type="int8", always_ram=False)  # on-disk quantization dei vettori
            ),
        )

        # Indice full-text sul campo 'text' per filtri MatchText
        self.client.create_payload_index(
            collection_name=self.settings.collection,
            field_name="text",
            field_schema=PayloadSchemaType.TEXT
        )

        # Indici keyword per filtri esatti / velocità nei filtri
        for key in ["doc_id", "source", "title", "lang"]:
            self.client.create_payload_index(
                collection_name=self.settings.collection,
                field_name=key,
                field_schema=PayloadSchemaType.KEYWORD
            )


    def build_points(self, chunks: List[Document], embeds: List[List[float]]) -> List[PointStruct]:
        pts: List[PointStruct] = []
        for i, (doc, vec) in enumerate(zip(chunks, embeds), start=1):
            payload = {
                "doc_id": doc.metadata.get("id"),
                "source": doc.metadata.get("source"),
                "title": doc.metadata.get("title"),
                "lang": doc.metadata.get("lang", "en"),
                "text": doc.page_content,
                "chunk_id": i - 1
            }
            pts.append(PointStruct(id=i, vector=vec, payload=payload))
        return pts
    
    def upsert_chunks(self, chunks: List[Document]):
        vecs = self.embeddings.embed_documents([c.page_content for c in chunks])
        points = self.build_points(chunks, vecs)
        self.client.upsert(collection_name=self.settings.collection, points=points, wait=True)