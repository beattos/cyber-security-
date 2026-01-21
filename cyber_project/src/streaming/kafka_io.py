from __future__ import annotations

import json
from typing import Any, Dict

from kafka import KafkaConsumer, KafkaProducer


def _json_serializer(value: Dict[str, Any]) -> bytes:
    return json.dumps(value).encode("utf-8")


def _json_deserializer(value: bytes) -> Dict[str, Any]:
    if value is None:
        return {}
    try:
        return json.loads(value.decode("utf-8"))
    except Exception:
        return {}


def build_producer(bootstrap_servers: str = "kafka:9092") -> KafkaProducer:
    """
    Build a Kafka producer for JSON messages.
    """
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=_json_serializer,
        key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
        linger_ms=5,
    )


def build_consumer(
    topic: str,
    bootstrap_servers: str = "kafka:9092",
    group_id: str = "cyber-consumer",
) -> KafkaConsumer:
    """
    Build a Kafka consumer for JSON messages.
    """
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=_json_deserializer,
    )
    return consumer


def validate_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize an incoming event.
    Returns the (possibly cleaned) event dict.
    Raises ValueError if required fields are missing / invalid.
    """
    if not isinstance(event, dict):
        raise ValueError("Event must be a JSON object")

    sample_id = event.get("sample_id")
    source_type = event.get("source_type")
    features = event.get("features")

    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Event missing valid 'sample_id'")
    if source_type not in ("static", "dynamic"):
        raise ValueError("Event 'source_type' must be 'static' or 'dynamic'")
    if not isinstance(features, dict):
        raise ValueError("Event 'features' must be an object/dict")

    # metadata is optional
    metadata = event.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"_invalid_metadata": True}

    return {
        "sample_id": sample_id,
        "source_type": source_type,
        "features": features,
        "metadata": metadata,
    }

