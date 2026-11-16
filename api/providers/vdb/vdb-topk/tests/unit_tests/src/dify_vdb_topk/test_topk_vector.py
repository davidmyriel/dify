import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from dify_vdb_topk import topk_vector as module
from dify_vdb_topk.topk_vector import TopKVector, TopKVectorConfig, TopKVectorFactory
from pydantic import ValidationError
from topk_sdk.error import CollectionAlreadyExistsError, CollectionNotFoundError, QueryLsnTimeoutError
from topk_sdk.schema import f32_vector, vector_index

from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.models.document import Document
from models.dataset import Dataset


def _config(**overrides: Any) -> TopKVectorConfig:
    values: dict[str, Any] = {"api_key": "key", "region": "aws-us-east-1-elastica"}
    values.update(overrides)
    return TopKVectorConfig.model_validate(values)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock(name="Client")
    fake.collection.return_value.upsert.return_value = "lsn-1"
    fake.collection.return_value.delete.return_value = "lsn-2"
    fake.collection.return_value.query.return_value = []
    # The class itself, so tests can check how the client was constructed.
    fake.constructor = MagicMock(return_value=fake)
    monkeypatch.setattr(module, "Client", fake.constructor)
    return fake


@pytest.fixture
def redis(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock(name="redis_client")
    fake.get.return_value = None
    monkeypatch.setattr(module, "redis_client", fake)
    return fake


@pytest.fixture
def vector(client: MagicMock, redis: MagicMock) -> TopKVector:
    return TopKVector("Vector_index_abc_Node", _config())


def _doc(doc_id: str, document_id: str = "document-1", text: str = "hello world") -> Document:
    return Document(
        page_content=text,
        metadata={"doc_id": doc_id, "doc_hash": doc_id, "document_id": document_id, "dataset_id": "dataset-1"},
    )


@pytest.mark.parametrize("missing", ["api_key", "region"])
def test_config_requires_credentials(missing: str):
    with pytest.raises(ValidationError, match=f"TOPK_{missing.upper()} is required"):
        _config(**{missing: ""})


def test_config_rejects_unknown_metric():
    with pytest.raises(ValidationError):
        _config(distance_metric="manhattan")


def test_get_type(vector: TopKVector):
    assert vector.get_type() == VectorType.TOPK == "topk"


def test_create_makes_collection_with_bm25_and_vector_index(vector: TopKVector, client: MagicMock, redis: MagicMock):
    ids = vector.create([_doc("a")], [[0.1, 0.2, 0.3]])

    assert ids == ["a"]
    (name,) = client.collections.return_value.create.call_args.args
    schema = client.collections.return_value.create.call_args.kwargs["schema"]
    assert name == "Vector_index_abc_Node"
    assert "KeywordIndex" in str(schema["text"])
    assert "F32Vector { dimension: 3 }" in str(schema["vector"])
    assert "metric: Cosine" in str(schema["vector"])
    redis.set.assert_any_call("vector_indexing_Vector_index_abc_Node", 1, ex=3600)


def test_create_skips_when_cached(vector: TopKVector, client: MagicMock, redis: MagicMock):
    redis.get.side_effect = lambda key: 1 if key.startswith("vector_indexing_") else None

    vector.create([_doc("a")], [[0.1, 0.2]])

    client.collections.return_value.create.assert_not_called()
    client.collection.return_value.upsert.assert_called_once()


def test_create_accepts_existing_compatible_collection(vector: TopKVector, client: MagicMock):
    collections = client.collections.return_value
    collections.create.side_effect = CollectionAlreadyExistsError("exists")
    collections.get.return_value.schema = {"vector": f32_vector(2).index(vector_index(metric="cosine"))}

    vector.create([_doc("a")], [[0.1, 0.2]])

    client.collection.return_value.upsert.assert_called_once()


def test_create_rejects_existing_collection_with_other_dimension(vector: TopKVector, client: MagicMock):
    collections = client.collections.return_value
    collections.create.side_effect = CollectionAlreadyExistsError("exists")
    collections.get.return_value.schema = {"vector": f32_vector(1536).index(vector_index(metric="cosine"))}

    with pytest.raises(ValueError, match="2-dimension vector index"):
        vector.create([_doc("a")], [[0.1, 0.2]])


def test_create_waits_out_recent_delete(vector: TopKVector, redis: MagicMock, monkeypatch: pytest.MonkeyPatch):
    now = 1_000_000.0
    redis.get.side_effect = lambda key: str(int(now) - 5) if key.startswith("topk_collection_deleted_") else None
    monkeypatch.setattr(module.time, "time", lambda: now)
    sleep = MagicMock()
    monkeypatch.setattr(module.time, "sleep", sleep)

    vector.create([_doc("a")], [[0.1]])

    sleep.assert_called_once_with(pytest.approx(module.RECREATE_GRACE_SECONDS - 5))


def test_add_texts_writes_rows_and_waits_for_lsn(vector: TopKVector, client: MagicMock):
    ids = vector.add_texts([_doc("a"), _doc("b", document_id="document-2")], [[1, 2], [3, 4]])

    assert ids == ["a", "b"]
    rows = client.collection.return_value.upsert.call_args.args[0]
    assert [r["_id"] for r in rows] == ["a", "b"]
    assert rows[0]["text"] == "hello world"
    assert rows[0]["vector"] == [1.0, 2.0]
    assert all(isinstance(x, float) for x in rows[0]["vector"])
    assert rows[1]["metadata"]["document_id"] == "document-2"
    assert json.loads(rows[1]["metadata_json"]) == rows[1]["metadata"]
    client.collection.return_value.count.assert_called_once_with(lsn="lsn-1")


def test_add_texts_generates_id_without_doc_id(vector: TopKVector, client: MagicMock):
    (doc_id,) = vector.add_texts([Document(page_content="x", metadata={})], [[1.0]]) or []

    assert doc_id
    assert client.collection.return_value.upsert.call_args.args[0][0]["_id"] == doc_id


def test_add_texts_rejects_mismatched_embeddings(vector: TopKVector):
    with pytest.raises(ValueError, match="2 documents but 1 embeddings"):
        vector.add_texts([_doc("a"), _doc("b")], [[1.0]])


def test_add_texts_skips_lsn_wait_when_disabled(client: MagicMock, redis: MagicMock):
    vector = TopKVector("c", _config(wait_for_indexing=False))
    vector.add_texts([_doc("a")], [[1.0]])

    client.collection.return_value.count.assert_not_called()


def test_lsn_wait_retries_then_raises(vector: TopKVector, client: MagicMock):
    client.collection.return_value.count.side_effect = QueryLsnTimeoutError("timeout")

    with pytest.raises(QueryLsnTimeoutError):
        vector.add_texts([_doc("a")], [[1.0]])
    assert client.collection.return_value.count.call_count == module.LSN_WAIT_ATTEMPTS


def test_upserts_are_split_by_request_size(vector: TopKVector, client: MagicMock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 400)
    client.collection.return_value.upsert.side_effect = ["lsn-a", "lsn-b", "lsn-c"]

    vector.add_texts([_doc(str(i), text="x" * 100) for i in range(3)], [[1.0]] * 3)

    assert client.collection.return_value.upsert.call_count == 3
    client.collection.return_value.count.assert_called_once_with(lsn="lsn-c")


def test_search_by_vector_builds_one_filtered_query(vector: TopKVector, client: MagicMock):
    client.collection.return_value.query.return_value = [
        {"_id": "a", "text": "hello", "metadata_json": json.dumps({"doc_id": "a"}), "score": 0.9},
        {"_id": "b", "text": "low", "metadata_json": json.dumps({"doc_id": "b"}), "score": 0.2},
    ]

    docs = vector.search_by_vector([0.1, 0.2], top_k=3, score_threshold=0.5, document_ids_filter=["d1", "d2"])

    assert [(d.metadata["doc_id"], d.metadata["score"]) for d in docs] == [("a", 0.9)]
    query = repr(client.collection.return_value.query.call_args.args[0])
    assert 'VectorScore { field: "vector"' in query
    assert 'field(metadata.document_id), op=In, right=literal(List(List { values: String(["d1", "d2"]) }))' in query
    assert "order: Desc" in query
    assert "Limit { k: 3 }" in query


def test_search_by_vector_euclidean_sorts_ascending_and_inverts_score(client: MagicMock, redis: MagicMock):
    vector = TopKVector("c", _config(distance_metric="euclidean"))
    client.collection.return_value.query.return_value = [{"text": "t", "metadata_json": "{}", "score": 1.0}]

    (doc,) = vector.search_by_vector([0.1])

    assert doc.metadata["score"] == pytest.approx(0.5)
    assert "order: Asc" in repr(client.collection.return_value.query.call_args.args[0])


def test_top_k_is_capped_at_query_limit(vector: TopKVector, client: MagicMock):
    vector.search_by_vector([0.1], top_k=50_000)

    assert "Limit { k: 10000 }" in repr(client.collection.return_value.query.call_args.args[0])


def test_search_by_full_text_uses_bm25(vector: TopKVector, client: MagicMock):
    client.collection.return_value.query.return_value = [
        {"_id": "a", "text": "hello", "metadata_json": json.dumps({"doc_id": "a"}), "score": 7.5}
    ]

    (doc,) = vector.search_by_full_text('say \\"hello\\"', top_k=2, document_ids_filter=["d1"])

    assert doc.metadata == {"doc_id": "a", "score": 7.5}
    query = repr(client.collection.return_value.query.call_args.args[0])
    assert "KeywordScore" in query
    assert 'token: "say \\"hello\\""' in query
    assert 'field: Some("text")' in query
    assert "metadata.document_id" in query


def test_search_by_full_text_ignores_blank_query(vector: TopKVector, client: MagicMock):
    assert vector.search_by_full_text("   ") == []
    client.collection.return_value.query.assert_not_called()


def test_reads_on_missing_collection_return_nothing(vector: TopKVector, client: MagicMock):
    client.collection.return_value.query.side_effect = CollectionNotFoundError("missing")
    client.collection.return_value.get.side_effect = CollectionNotFoundError("missing")

    assert vector.search_by_vector([0.1]) == []
    assert vector.search_by_full_text("hello") == []
    assert vector.text_exists("a") is False
    assert vector.get_ids_by_metadata_field("app_id", "x") is None


def test_reads_use_own_lsn_after_write(vector: TopKVector, client: MagicMock):
    client.collection.return_value.get.return_value = {"a": {}}
    vector.add_texts([_doc("a")], [[1.0]])

    assert vector.text_exists("a") is True
    client.collection.return_value.get.assert_called_once_with(["a"], fields=[], lsn="lsn-1")


def test_reads_use_configured_consistency(client: MagicMock, redis: MagicMock):
    vector = TopKVector("c", _config(consistency="strong"))
    vector.search_by_full_text("hello")

    assert client.collection.return_value.query.call_args.kwargs == {"consistency": "strong"}


def test_reads_default_to_balanced_consistency(vector: TopKVector, client: MagicMock):
    vector.search_by_full_text("hello")

    assert client.collection.return_value.query.call_args.kwargs == {}


def test_get_ids_by_metadata_field(vector: TopKVector, client: MagicMock):
    client.collection.return_value.query.return_value = [{"_id": "a"}, {"_id": "b"}]

    assert vector.get_ids_by_metadata_field("annotation_id", "x") == ["a", "b"]
    query = repr(client.collection.return_value.query.call_args.args[0])
    assert 'field(metadata.annotation_id), op=Eq, right=literal(String("x"))' in query


def test_delete_by_ids(vector: TopKVector, client: MagicMock):
    vector.delete_by_ids(["a", "b"])

    client.collection.return_value.delete.assert_called_once_with(["a", "b"])
    client.collection.return_value.count.assert_called_once_with(lsn="lsn-2")


def test_delete_by_ids_noop_for_empty_list(vector: TopKVector, client: MagicMock):
    vector.delete_by_ids([])

    client.collection.return_value.delete.assert_not_called()


def test_delete_by_metadata_field_uses_filter(vector: TopKVector, client: MagicMock):
    vector.delete_by_metadata_field("app_id", "app-1")

    (expr,) = client.collection.return_value.delete.call_args.args
    assert repr(expr) == 'Binary(left=field(metadata.app_id), op=Eq, right=literal(String("app-1")))'


def test_deletes_on_missing_collection_are_noops(vector: TopKVector, client: MagicMock, redis: MagicMock):
    client.collection.return_value.delete.side_effect = CollectionNotFoundError("missing")
    client.collections.return_value.delete.side_effect = CollectionNotFoundError("missing")

    vector.delete_by_ids(["a"])
    vector.delete_by_metadata_field("app_id", "x")
    vector.delete()

    redis.set.assert_not_called()


def test_delete_drops_collection_and_records_tombstone(vector: TopKVector, client: MagicMock, redis: MagicMock):
    vector.delete()

    client.collections.return_value.delete.assert_called_once_with("Vector_index_abc_Node")
    redis.set.assert_called_once()
    assert redis.set.call_args.args[0] == "topk_collection_deleted_Vector_index_abc_Node"
    assert redis.set.call_args.kwargs == {"ex": module.RECREATE_GRACE_SECONDS}
    redis.delete.assert_called_once_with("vector_indexing_Vector_index_abc_Node")


def _dataset(index_struct: str | None = None) -> Dataset:
    return Dataset(
        id="11111111-2222-3333-4444-555555555555",
        tenant_id="tenant",
        name="d",
        data_source_type="upload_file",
        created_by="user",
        index_struct=index_struct,
    )


def test_factory_creates_index_struct_for_new_dataset(client: MagicMock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(module.dify_config, "TOPK_API_KEY", "key")
    monkeypatch.setattr(module.dify_config, "TOPK_REGION", "aws-us-east-1-elastica")
    dataset = _dataset()

    vector = TopKVectorFactory().init_vector(dataset, [], MagicMock(), session=None)

    assert vector.collection_name == Dataset.gen_collection_name_by_id(dataset.id)
    assert json.loads(dataset.index_struct or "")["type"] == "topk"
    client.constructor.assert_called_once_with(
        api_key="key", region="aws-us-east-1-elastica", host="topk.io", https=True
    )


def test_factory_reuses_existing_collection_name(client: MagicMock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(module.dify_config, "TOPK_API_KEY", "key")
    monkeypatch.setattr(module.dify_config, "TOPK_REGION", "r")
    dataset = _dataset(json.dumps({"type": "topk", "vector_store": {"class_prefix": "Existing_Node"}}))

    vector = TopKVectorFactory().init_vector(dataset, [], MagicMock(), session=None)

    assert vector.collection_name == "Existing_Node"


def test_factory_requires_api_key(client: MagicMock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(module.dify_config, "TOPK_API_KEY", None)
    monkeypatch.setattr(module.dify_config, "TOPK_REGION", "r")

    with pytest.raises(ValidationError, match="TOPK_API_KEY is required"):
        TopKVectorFactory().init_vector(_dataset(), [], MagicMock(), session=None)
