"""End-to-end tests against a real TopK project.

Set ``TOPK_API_KEY`` (and optionally ``TOPK_REGION``) to run them; they are skipped
otherwise. Every test creates collections with fresh UUID names and deletes them.
"""

import os
import uuid
from typing import override

import pytest
from dify_vdb_topk.topk_vector import TopKVector, TopKVectorConfig

from core.rag.datasource.vdb.vector_integration_test_support import AbstractVectorTest
from core.rag.models.document import Document

pytestmark = pytest.mark.skipif(not os.getenv("TOPK_API_KEY"), reason="TOPK_API_KEY is not set")


def _config() -> TopKVectorConfig:
    return TopKVectorConfig(
        api_key=os.environ.get("TOPK_API_KEY", ""),
        region=os.environ.get("TOPK_REGION", "aws-us-east-1-elastica"),
    )


def _doc(text: str, document_id: str, **extra: str) -> Document:
    doc_id = str(uuid.uuid4())
    return Document(
        page_content=text,
        metadata={"doc_id": doc_id, "doc_hash": doc_id, "document_id": document_id, "dataset_id": "e2e", **extra},
    )


class TopKVectorTest(AbstractVectorTest):
    def __init__(self):
        super().__init__()
        self.vector = TopKVector(collection_name=self.collection_name, config=_config())

    @override
    def get_ids_by_metadata_field(self):
        ids = self.vector.get_ids_by_metadata_field(key="document_id", value=self.example_doc_id)
        assert ids == [self.example_doc_id]

    def search_by_vector_scores(self):
        (hit,) = self.vector.search_by_vector(query_vector=self.example_embedding, score_threshold=0.99)
        # Cosine similarity of a vector with itself.
        assert hit.metadata["score"] == pytest.approx(1.0, abs=1e-3)


@pytest.fixture
def cleanup():
    vectors: list[TopKVector] = []
    yield vectors
    for vector in vectors:
        vector.delete()


def test_topk_vector(setup_mock_redis, cleanup):
    test = TopKVectorTest()
    cleanup.append(test.vector)
    test.create_vector()
    test.search_by_vector_scores()
    test.run_all_tests()


def test_topk_bm25_filters_and_metadata_deletes(setup_mock_redis, cleanup):
    vector = TopKVector(collection_name=f"Vector_index_{uuid.uuid4().hex}_Node_test", config=_config())
    cleanup.append(vector)
    docs = [
        _doc("TopK runs BM25 keyword search natively", "doc-a", app_id="app-1"),
        _doc("Partitions isolate tenants in one collection", "doc-b", app_id="app-1"),
        _doc("Read consistency can be balanced, indexed or strong", "doc-c", app_id="app-2"),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    vector.create(docs, embeddings)

    # A fresh client has no LSN: this is what a Dify API process sees right after
    # the indexing worker returns.
    reader = TopKVector(collection_name=vector.collection_name, config=_config())

    hits = reader.search_by_full_text("keyword search", top_k=3)
    assert [h.metadata["document_id"] for h in hits] == ["doc-a"]
    assert hits[0].metadata["score"] > 0

    assert reader.search_by_full_text("tenants", document_ids_filter=["doc-a", "doc-c"]) == []
    hits = reader.search_by_vector([0.0, 1.0, 0.0], top_k=3, document_ids_filter=["doc-a", "doc-c"])
    assert {h.metadata["document_id"] for h in hits} == {"doc-a", "doc-c"}
    hits = reader.search_by_vector([0.0, 1.0, 0.0], top_k=1)
    assert hits[0].metadata["document_id"] == "doc-b"
    assert hits[0].metadata["app_id"] == "app-1"

    vector.delete_by_metadata_field("app_id", "app-1")
    remaining = reader.search_by_vector([1.0, 1.0, 1.0], top_k=10)
    assert [h.metadata["document_id"] for h in remaining] == ["doc-c"]
