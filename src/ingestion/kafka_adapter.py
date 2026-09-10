"""
Kafka Adapter (optional)
========================
Producer / Consumer wrappers around the stream simulator.
Requires: pip install confluent-kafka

Producer: feeds events from StreamSimulator → Kafka topic
Consumer: reads from Kafka topic → yields event dicts for the pipeline

Set kafka_enabled: true in config/settings.yaml to activate.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Generator, Iterator

import yaml

log = logging.getLogger(__name__)

try:
    from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    log.warning(
        "confluent-kafka not installed. Kafka adapter is disabled. "
        "Install with: pip install confluent-kafka"
    )


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _event_to_json(event: dict) -> bytes:
    """Serialise an event dict to JSON bytes, handling datetime objects."""
    evt = event.copy()
    if isinstance(evt.get("timestamp"), datetime):
        evt["timestamp"] = evt["timestamp"].isoformat()
    return json.dumps(evt).encode("utf-8")


def _json_to_event(data: bytes) -> dict:
    evt = json.loads(data.decode("utf-8"))
    if "timestamp" in evt and isinstance(evt["timestamp"], str):
        evt["timestamp"] = datetime.fromisoformat(evt["timestamp"])
    return evt


# ---------------------------------------------------------------------------
# Kafka Producer
# ---------------------------------------------------------------------------

class PatientEventProducer:
    """
    Wraps a StreamSimulator and publishes events to a Kafka topic.

    Usage:
        producer = PatientEventProducer()
        producer.run(simulator, n_stays=200)
    """

    def __init__(self, cfg_path: str = "config/settings.yaml"):
        if not KAFKA_AVAILABLE:
            raise RuntimeError("confluent-kafka is not installed.")
        cfg = _load_cfg(cfg_path)
        self.topic = cfg["streaming"]["kafka_topic"]
        bootstrap = cfg["streaming"]["kafka_bootstrap_servers"]
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap,
                "queue.buffering.max.ms": 50,
                "batch.num.messages": 100,
            }
        )
        log.info(f"Kafka producer connected to {bootstrap}, topic={self.topic}")

    def _delivery_report(self, err, msg):
        if err:
            log.error(f"Kafka delivery failed: {err}")

    def publish(self, event: dict) -> None:
        key = (event.get("patient_id") or "unknown").encode("utf-8")
        self._producer.produce(
            self.topic,
            key=key,
            value=_event_to_json(event),
            callback=self._delivery_report,
        )
        self._producer.poll(0)  # trigger delivery callbacks without blocking

    def flush(self) -> None:
        self._producer.flush()

    def run(self, simulator, n_stays: int | None = None, realtime: bool = True) -> None:
        """Publish all events from the simulator to Kafka."""
        published = 0
        try:
            for event in simulator.stream(n_stays=n_stays, realtime=realtime):
                self.publish(event)
                published += 1
                if published % 10_000 == 0:
                    log.info(f"Published {published:,} events to Kafka")
        finally:
            self.flush()
            log.info(f"Kafka producer finished. Total published: {published:,}")


# ---------------------------------------------------------------------------
# Kafka Consumer
# ---------------------------------------------------------------------------

class PatientEventConsumer:
    """
    Reads events from a Kafka topic and yields event dicts.

    Usage:
        consumer = PatientEventConsumer()
        for event in consumer.consume():
            pipeline.process(event)
    """

    def __init__(self, cfg_path: str = "config/settings.yaml", group_id: str = "ews-pipeline"):
        if not KAFKA_AVAILABLE:
            raise RuntimeError("confluent-kafka is not installed.")
        cfg = _load_cfg(cfg_path)
        self.topic = cfg["streaming"]["kafka_topic"]
        bootstrap = cfg["streaming"]["kafka_bootstrap_servers"]
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
                "max.poll.interval.ms": 300_000,
            }
        )
        self._consumer.subscribe([self.topic])
        log.info(f"Kafka consumer subscribed to {self.topic}")

    def consume(self, timeout: float = 1.0) -> Generator[dict, None, None]:
        """Continuously yield events from Kafka until interrupted."""
        try:
            while True:
                msg = self._consumer.poll(timeout)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise KafkaException(msg.error())
                yield _json_to_event(msg.value())
        finally:
            self._consumer.close()
            log.info("Kafka consumer closed.")

    def consume_batch(
        self, batch_size: int = 10, timeout: float = 1.0
    ) -> Generator[list[dict], None, None]:
        """Yield events in batches from Kafka."""
        batch = []
        for event in self.consume(timeout=timeout):
            batch.append(event)
            if len(batch) >= batch_size:
                yield batch
                batch = []
