from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class TopKConfig(BaseSettings):
    """
    Configuration settings for TopK (https://topk.io).

    Create an API key in the TopK console (https://console.topk.io). A key is scoped
    to one project, so no project ID is needed.
    """

    TOPK_API_KEY: str | None = Field(
        description="TopK API key, from https://console.topk.io.",
        default=None,
    )

    TOPK_REGION: str | None = Field(
        description="TopK region of the project, e.g. 'aws-us-east-1-elastica'. See https://docs.topk.io/regions.",
        default=None,
    )

    TOPK_HOST: str = Field(
        description="TopK API host. Only change this for a dedicated TopK deployment.",
        default="topk.io",
    )

    TOPK_HTTPS: bool = Field(
        description="Whether to connect to TopK over TLS.",
        default=True,
    )

    TOPK_DISTANCE_METRIC: Literal["cosine", "dot_product", "euclidean"] = Field(
        description="Vector index metric for new collections: 'cosine', 'dot_product' or 'euclidean'.",
        default="cosine",
    )

    TOPK_CONSISTENCY: Literal["indexed", "strong"] | None = Field(
        description="Read consistency. Leave empty for TopK's default balanced mode;"
        " 'strong' always reads the latest writes, 'indexed' trades freshness for latency."
        " See https://docs.topk.io/concepts#read-consistency.",
        default=None,
    )

    TOPK_WAIT_FOR_INDEXING: bool = Field(
        description="Wait until written chunks are queryable before indexing finishes,"
        " so documents can be retrieved as soon as Dify marks them available.",
        default=True,
    )
