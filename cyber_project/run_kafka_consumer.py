import json
import logging
import os
from typing import Any, Dict

import pandas as pd

from src.agents.common.contract import Sample
from src.agents.common.utils import load_policy_config
from src.agents.enforcing.enforce import ConfidenceEnforcingAgent
from src.agents.inference.ada_dynamic import AdaDynamicAgent
from src.agents.inference.ada_static import AdaStaticAgent
from src.agents.inference.gb_dynamic import GBDynamicAgent
from src.agents.inference.gb_static import GBStaticAgent
from src.pipeline.orchestrator import Orchestrator
from src.streaming.kafka_io import build_consumer, build_producer, validate_event


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kafka-consumer")


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_orchestrator() -> Orchestrator:
    static_order = load_json("artifacts/static_feature_order.json")
    dynamic_order = load_json("artifacts/dynamic_feature_order.json")
    policy = load_policy_config()

    enforcing_agent = ConfidenceEnforcingAgent(
        static_t_high=policy["static_t_high"],
        static_t_low=policy["static_t_low"],
        dynamic_t_high=policy["dynamic_t_high"],
        dynamic_t_low=policy["dynamic_t_low"],
        disagreement_penalty=policy["disagreement_penalty"],
        impute_penalty_mid=policy["impute_penalty_mid"],
        impute_penalty_high=policy["impute_penalty_high"],
    )

    agents = [
        AdaStaticAgent(
            "ada_static",
            "models/ada_static.pkl",
            static_order,
            mal_threshold=policy["static_mal_threshold"],
        ),
        GBStaticAgent(
            "gb_static",
            "models/gb_static.pkl",
            static_order,
            mal_threshold=policy["static_mal_threshold"],
        ),
        AdaDynamicAgent(
            "ada_dynamic",
            "models/ada_dynamic_calibrated.pkl",
            dynamic_order,
            mal_threshold=policy["dynamic_mal_threshold"],
        ),
        GBDynamicAgent(
            "gb_dynamic",
            "models/gb_dynamic_calibrated.pkl",
            dynamic_order,
            mal_threshold=policy["dynamic_mal_threshold"],
        ),
    ]

    return Orchestrator(agents, enforcing_agent=enforcing_agent)


def event_to_sample(event: Dict[str, Any]) -> Sample:
    """
    Convert validated event dict into Sample.
    """
    return Sample(
        sample_id=event["sample_id"],
        source_type=event["source_type"],
        features=event["features"],
        metadata=event.get("metadata") or {},
        label=None,
    )


def main():
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    input_topic = os.getenv("KAFKA_INPUT_TOPIC", "malware-input")
    output_topic = os.getenv("KAFKA_OUTPUT_TOPIC", "malware-decisions")
    group_id = os.getenv("KAFKA_CONSUMER_GROUP", "cyber-consumer")

    logger.info(
        "Starting Kafka consumer: bootstrap=%s, input_topic=%s, output_topic=%s, group_id=%s",
        bootstrap,
        input_topic,
        output_topic,
        group_id,
    )

    orch = build_orchestrator()
    consumer = build_consumer(input_topic, bootstrap_servers=bootstrap, group_id=group_id)
    producer = build_producer(bootstrap_servers=bootstrap)

    try:
        for msg in consumer:
            try:
                event = msg.value
                event = validate_event(event)
            except Exception as e:
                logger.warning("Invalid event on topic %s, partition %s, offset %s: %s",
                               msg.topic, msg.partition, msg.offset, e)
                # Skip invalid message; do not commit to allow potential re-processing
                continue

            sample = event_to_sample(event)

            try:
                decision = orch.run(sample)
                out = decision.model_dump()
                producer.send(
                    output_topic,
                    key=sample.sample_id,
                    value=out,
                )
                producer.flush()
                consumer.commit()
                logger.info(
                    "Processed sample_id=%s action=%s chosen_agent=%s effective_confidence=%.4f",
                    decision.sample_id,
                    decision.action,
                    decision.judgement.chosen_agent,
                    decision.effective_confidence,
                )
            except Exception as e:
                logger.exception("Error processing sample_id=%s: %s", sample.sample_id, e)
                # Do not commit offset so message can be retried

    except KeyboardInterrupt:
        logger.info("Shutting down consumer...")
    finally:
        try:
            consumer.close()
        except Exception:
            pass
        try:
            producer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

