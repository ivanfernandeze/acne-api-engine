"""In-process latency and error telemetry.

Feeds ``GET /audit/iia-check`` (RF-TR-04) and the RNF-01 latency budget. Kept in memory on
purpose: the durable record of every inference already lives in ``ai_inferences`` in the clinical
database, and duplicating it here would create a second, divergent source of truth. What the
engine adds is its OWN view of the last 24 hours, independent of the orchestrator.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Observation:
    timestamp: float
    latency_ms: float
    failed: bool


@dataclass
class Telemetry:
    """Rolling 24-hour window of inference outcomes."""

    window_seconds: float = 24 * 60 * 60
    started_at: float = field(default_factory=time.time)
    _observations: list[_Observation] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, latency_ms: float, failed: bool = False) -> None:
        with self._lock:
            self._observations.append(_Observation(time.time(), latency_ms, failed))
            self._prune()

    def snapshot(self) -> dict[str, float | int | None]:
        with self._lock:
            self._prune()
            latencies = sorted(o.latency_ms for o in self._observations if not o.failed)
            total = len(self._observations)
            failures = sum(1 for o in self._observations if o.failed)

        return {
            "count": total,
            "median_ms": _quantile(latencies, 0.5),
            "p95_ms": _quantile(latencies, 0.95),
            "error_rate": 0.0 if total == 0 else failures / total,
            "uptime_hours": (time.time() - self.started_at) / 3600.0,
        }

    def _prune(self) -> None:
        cutoff = time.time() - self.window_seconds
        self._observations = [o for o in self._observations if o.timestamp >= cutoff]


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(len(sorted_values) * q))
    return round(sorted_values[index], 2)


telemetry = Telemetry()
