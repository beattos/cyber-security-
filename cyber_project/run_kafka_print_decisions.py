import json
import logging
import os

from src.streaming.kafka_io import build_consumer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kafka-decisions")


def main():
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    topic = os.getenv("KAFKA_OUTPUT_TOPIC", "malware-decisions")
    group_id = os.getenv("KAFKA_DECISIONS_GROUP", "cyber-decisions-viewer")

    consumer = build_consumer(topic, bootstrap_servers=bootstrap, group_id=group_id)
    logger.info("Consuming decisions from topic=%s bootstrap=%s", topic, bootstrap)

    try:
        for msg in consumer:
            val = msg.value
            if not isinstance(val, dict):
                try:
                    val = json.loads(val)
                except Exception:
                    logger.warning("Skipping non-JSON message at offset %s", msg.offset)
                    consumer.commit()
                    continue

            sample_id = val.get("sample_id")
            action = val.get("action")
            eff = val.get("effective_confidence")
            judgement = val.get("judgement", {}) or {}
            chosen_agent = judgement.get("chosen_agent")

            print(
                f"sample_id={sample_id} action={action} "
                f"effective_confidence={eff} chosen_agent={chosen_agent}"
            )
            consumer.commit()
    except KeyboardInterrupt:
        logger.info("Stopping decision consumer...")
    finally:
        try:
            consumer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

