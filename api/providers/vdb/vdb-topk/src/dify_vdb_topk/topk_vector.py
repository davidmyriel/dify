"""TopK vector store backend.

TopK (https://topk.io) is a hosted search database. This backend maps Dify's
retrieval methods onto TopK's native features:

* semantic search: ANN over the embeddings Dify computes, in a ``vector_index``;
* full-text search: native BM25 (``keyword_index`` + ``match`` + ``bm25_score``);
* ``document_ids_filter``: a filter expression evaluated inside the same query.

Each Dify collection (one per dataset, or one per annotation binding) is one
TopK collection, so datasets with different embedding models and dimensions
never share a schema.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator, Sequence
from typing import Any, Literal, override

from pydantic import BaseModel, model_validator
from sqlalchemy.orm import Session
from topk_sdk import Client
from topk_sdk.error import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    QueryLsnTimeoutError,
)
from topk_sdk.query import field, fn, match, select
from topk_sdk.schema import f32_vector, keyword_index, text, vector_index

from configs import dify_config
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.embedding.embedding_base import Embeddings
from core.rag.models.document import Document
from extensions.ext_redis import redis_client
from models.dataset import Dataset

logger = logging.getLogger(__name__)

Metric = Literal["cosine", "dot_product", "euclidean"]
Consistency = Literal["indexed", "strong"]

TEXT_FIELD = "text"
VECTOR_FIELD = "vector"
# Filters run against the ``metadata`` struct (``metadata.document_id``). A query
# can't return a whole struct, so results read the JSON copy instead.
METADATA_FIELD = "metadata"
METADATA_JSON_FIELD = "metadata_json"
# TopK SQL rejects aliases that start with an underscore.
SCORE_ALIAS = "score"

MAX_QUERY_RESULTS = 10_000
"""TopK returns at most this many results from one query."""

MAX_REQUEST_BYTES = 4 * 1024 * 1024
"""Target upsert payload size, well under TopK's 8MB request limit."""

RECREATE_GRACE_SECONDS = 65
"""How long to wait before recreating a collection under a just-deleted name.

For about 60 seconds after a collection is deleted, writes to a new collection
with the same name can still be routed to the deleted one and are lost. Dify
reuses collection names (for example when a dataset's embedding model changes,
or annotation reply is turned off and on), so ``delete`` records a tombstone
in Redis and ``create`` waits it out.
"""

LSN_WAIT_ATTEMPTS = 5


class TopKVectorConfig(BaseModel):
    api_key: str
    region: str
    host: str = "topk.io"
    https: bool = True
    distance_metric: Metric = "cosine"
    consistency: Consistency | None = None
    wait_for_indexing: bool = True

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values.get("api_key"):
            raise ValueError("config TOPK_API_KEY is required")
        if not values.get("region"):
            raise ValueError("config TOPK_REGION is required")
        return values


class TopKVector(BaseVector):
    def __init__(self, collection_name: str, config: TopKVectorConfig):
        super().__init__(collection_name)
        self._config = config
        self._client = Client(
            api_key=config.api_key,
            region=config.region,
            host=config.host,
            https=config.https,
        )
        # LSN of this instance's last write, so its own reads include it.
        self._last_lsn: str | None = None

    @override
    def get_type(self) -> str:
        return VectorType.TOPK

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    @override
    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs: Any) -> list[str] | None:
        if not texts or not embeddings:
            return []
        self._create_collection(len(embeddings[0]))
        return self.add_texts(texts, embeddings, **kwargs)

    @override
    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs: Any) -> list[str] | None:
        if not documents:
            return []
        if len(documents) != len(embeddings):
            raise ValueError(f"Got {len(documents)} documents but {len(embeddings)} embeddings.")

        ids: list[str] = []
        rows: list[dict[str, Any]] = []
        for doc, embedding in zip(documents, embeddings, strict=True):
            metadata = dict(doc.metadata or {})
            doc_id = str(metadata.get("doc_id") or uuid.uuid4())
            ids.append(doc_id)
            rows.append(
                {
                    "_id": doc_id,
                    TEXT_FIELD: doc.page_content,
                    VECTOR_FIELD: [float(x) for x in embedding],
                    METADATA_FIELD: metadata,
                    METADATA_JSON_FIELD: json.dumps(metadata, default=str, ensure_ascii=False),
                }
            )

        collection = self._client.collection(self._collection_name)
        for batch in _size_batches(rows):
            self._last_lsn = collection.upsert(batch)
        self._wait_until_indexed()
        return ids

    @override
    def delete_by_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            self._last_lsn = self._client.collection(self._collection_name).delete(list(ids))
        except CollectionNotFoundError:
            return
        self._wait_until_indexed()

    @override
    def delete_by_metadata_field(self, key: str, value: str) -> None:
        try:
            self._last_lsn = self._client.collection(self._collection_name).delete(
                field(f"{METADATA_FIELD}.{key}").eq(value)
            )
        except CollectionNotFoundError:
            return
        self._wait_until_indexed()

    @override
    def delete(self) -> None:
        try:
            self._client.collections().delete(self._collection_name)
        except CollectionNotFoundError:
            pass
        else:
            redis_client.set(self._tombstone_key, int(time.time()), ex=RECREATE_GRACE_SECONDS)
        redis_client.delete(self._exists_cache_key)
        self._last_lsn = None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @override
    def text_exists(self, id: str) -> bool:
        try:
            found = self._client.collection(self._collection_name).get([id], fields=[], **self._read_options())
        except CollectionNotFoundError:
            return False
        return id in found

    @override
    def get_ids_by_metadata_field(self, key: str, value: str) -> list[str] | None:
        query = select().filter(field(f"{METADATA_FIELD}.{key}").eq(value)).limit(MAX_QUERY_RESULTS)
        rows = self._run(query)
        return [row["_id"] for row in rows] or None

    @override
    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        top_k = _top_k(kwargs)
        score_threshold = float(kwargs.get("score_threshold") or 0.0)
        euclidean = self._config.distance_metric == "euclidean"

        query = select(
            TEXT_FIELD,
            METADATA_JSON_FIELD,
            **{SCORE_ALIAS: fn.vector_distance(VECTOR_FIELD, [float(x) for x in query_vector])},
        )
        query = self._apply_document_filter(query, kwargs.get("document_ids_filter"))
        query = query.sort(field(SCORE_ALIAS), asc=euclidean).limit(top_k)

        docs: list[Document] = []
        for row in self._run(query):
            raw = float(row.get(SCORE_ALIAS) or 0.0)
            # Dify expects higher-is-better scores; euclidean returns a distance.
            score = 1.0 / (1.0 + raw) if euclidean else raw
            if score >= score_threshold:
                docs.append(_to_document(row, score))
        return docs

    @override
    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        # RetrievalService escapes double quotes for SQL-style backends; TopK
        # tokenizes the raw string itself.
        query = query.replace('\\"', '"').strip()
        if not query:
            return []
        top_k = _top_k(kwargs)

        topk_query = select(TEXT_FIELD, METADATA_JSON_FIELD, **{SCORE_ALIAS: fn.bm25_score()}).filter(
            match(query, field=TEXT_FIELD)
        )
        topk_query = self._apply_document_filter(topk_query, kwargs.get("document_ids_filter"))
        topk_query = topk_query.sort(field(SCORE_ALIAS), asc=False).limit(top_k)

        return [_to_document(row, float(row.get(SCORE_ALIAS) or 0.0)) for row in self._run(topk_query)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _tombstone_key(self) -> str:
        return f"topk_collection_deleted_{self._collection_name}"

    @property
    def _exists_cache_key(self) -> str:
        return f"vector_indexing_{self._collection_name}"

    def _create_collection(self, dimension: int) -> None:
        if redis_client.get(self._exists_cache_key):
            return
        self._wait_for_tombstone()
        lock_name = f"vector_indexing_lock_{self._collection_name}"
        with redis_client.lock(lock_name, timeout=20):
            if redis_client.get(self._exists_cache_key):
                return
            schema = {
                TEXT_FIELD: text().index(keyword_index()),
                VECTOR_FIELD: f32_vector(dimension).index(vector_index(metric=self._config.distance_metric)),
            }
            try:
                self._client.collections().create(self._collection_name, schema=schema)
            except CollectionAlreadyExistsError:
                self._check_existing_schema(dimension)
            redis_client.set(self._exists_cache_key, 1, ex=3600)

    def _wait_for_tombstone(self) -> None:
        deleted_at = redis_client.get(self._tombstone_key)
        if not deleted_at:
            return
        remaining = int(deleted_at) + RECREATE_GRACE_SECONDS - time.time()
        if remaining > 0:
            logger.info(
                "TopK collection %s was deleted recently; waiting %.0fs before recreating it",
                self._collection_name,
                remaining,
            )
            time.sleep(remaining)

    def _check_existing_schema(self, dimension: int) -> None:
        """Fail loudly if an existing collection can't hold these embeddings."""
        schema = self._client.collections().get(self._collection_name).schema
        # A field's dimension and metric are only visible in its text form.
        spec = str(schema.get(VECTOR_FIELD, ""))
        if "VectorIndex" not in spec or f"dimension: {dimension}" not in spec:
            raise ValueError(
                f"TopK collection {self._collection_name!r} exists with an incompatible "
                f"{VECTOR_FIELD!r} field ({spec or 'missing'}); expected a {dimension}-dimension vector index."
            )

    def _apply_document_filter(self, query: Any, document_ids: Sequence[str] | None) -> Any:
        if document_ids:
            query = query.filter(field(f"{METADATA_FIELD}.document_id").in_([str(i) for i in document_ids]))
        return query

    def _read_options(self) -> dict[str, Any]:
        if self._last_lsn:
            return {"lsn": self._last_lsn}
        if self._config.consistency:
            return {"consistency": self._config.consistency}
        return {}

    def _run(self, query: Any) -> list[dict[str, Any]]:
        try:
            return self._client.collection(self._collection_name).query(query, **self._read_options())
        except CollectionNotFoundError:
            return []

    def _wait_until_indexed(self) -> None:
        """Block until this instance's last write is queryable.

        Dify indexes in a worker and marks documents "available" when the
        worker returns; users often run a retrieval test right away from
        another process. Waiting on the write's LSN here makes "available"
        mean queryable, while reads stay on TopK's default consistency.
        """
        if not self._config.wait_for_indexing or not self._last_lsn:
            return
        collection = self._client.collection(self._collection_name)
        for attempt in range(1, LSN_WAIT_ATTEMPTS + 1):
            try:
                collection.count(lsn=self._last_lsn)
                return
            except QueryLsnTimeoutError:
                if attempt == LSN_WAIT_ATTEMPTS:
                    raise
                logger.info("TopK write %s not indexed yet, retrying (%d)", self._last_lsn, attempt)


def _top_k(kwargs: dict[str, Any]) -> int:
    top_k = int(kwargs.get("top_k") or 4)
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return min(top_k, MAX_QUERY_RESULTS)


def _to_document(row: dict[str, Any], score: float) -> Document:
    raw = row.get(METADATA_JSON_FIELD)
    metadata: dict[str, Any] = json.loads(raw) if raw else {}
    metadata["score"] = score
    return Document(page_content=row.get(TEXT_FIELD) or "", metadata=metadata)


def _size_batches(rows: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    """Split rows into upsert requests of roughly ``MAX_REQUEST_BYTES``."""
    batch: list[dict[str, Any]] = []
    size = 0
    for row in rows:
        row_size = (
            len(row[TEXT_FIELD].encode()) + 2 * len(row[METADATA_JSON_FIELD].encode()) + 8 * len(row[VECTOR_FIELD])
        )
        if batch and size + row_size > MAX_REQUEST_BYTES:
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += row_size
    if batch:
        yield batch


class TopKVectorFactory(AbstractVectorFactory):
    @override
    def init_vector(
        self, dataset: Dataset, attributes: list, embeddings: Embeddings, *, session: Session | None
    ) -> TopKVector:
        if dataset.index_struct_dict:
            collection_name: str = dataset.index_struct_dict["vector_store"]["class_prefix"]
        else:
            collection_name = Dataset.gen_collection_name_by_id(dataset.id)
            dataset.index_struct = json.dumps(self.gen_index_struct_dict(VectorType.TOPK, collection_name))

        return TopKVector(
            collection_name=collection_name,
            config=TopKVectorConfig(
                api_key=dify_config.TOPK_API_KEY or "",
                region=dify_config.TOPK_REGION or "",
                host=dify_config.TOPK_HOST,
                https=dify_config.TOPK_HTTPS,
                distance_metric=dify_config.TOPK_DISTANCE_METRIC,
                consistency=dify_config.TOPK_CONSISTENCY,
                wait_for_indexing=dify_config.TOPK_WAIT_FOR_INDEXING,
            ),
        )
