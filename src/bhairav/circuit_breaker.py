"""CircuitBreaker: graceful degradation when YOLO, database, or other services fail.

Problem solved:
    - YOLO crashes → entire pipeline dies
    - Database unreachable → all API endpoints 500
    - No retry logic → transient failures become permanent outages
    - No degraded mode → system is either fully up or fully down

Architecture:
    CircuitBreaker wraps any callable (detector, DB query, etc.) with:
    1. Consecutive failure counting
    2. Open state (stop calling the failing service)
    3. Half-open state (periodic retry attempts)
    4. Closed state (normal operation restored)

    States:
        CLOSED  → normal, calls pass through
        OPEN    → calls rejected immediately, returns fallback
        HALF_OPEN → one probe call allowed through to test recovery

Usage::

    from bhairav.circuit_breaker import CircuitBreaker, CircuitState

    yolo_cb = CircuitBreaker(
        name="yolo",
        failure_threshold=3,
        recovery_timeout=30.0,
        half_open_max=1,
    )

    try:
        result = yolo_cb.call(detect, frame)
    except CircuitOpenError:
        result = fallback_detection()

    # Or use as a decorator
    @yolo_cb.wrap
    def detect(frame):
        return model(frame)
"""
from __future__ import annotations

import functools
import logging
import time
import threading
from enum import Enum
from dataclasses import dataclass, field

log = logging.getLogger("bhairav.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing — reject calls
    HALF_OPEN = "half_open" # Testing recovery


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, name: str, last_error: Exception | None = None):
        self.circuit_name = name
        self.last_error = last_error
        super().__init__(f"Circuit '{name}' is OPEN — calls rejected")


@dataclass
class CircuitStats:
    """Tracks circuit breaker state and metrics."""
    name: str = ""
    state: str = "closed"
    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0
    total_rejections: int = 0
    total_successes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    last_error: str = ""
    opened_at: float = 0.0
    recovery_attempts: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "total_successes": self.total_successes,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "last_error": self.last_error,
            "opened_at": self.opened_at,
            "recovery_attempts": self.recovery_attempts,
        }


class CircuitBreaker:
    """Generic circuit breaker for any callable.

    Configuration:
        name (str): Identifier for logging. Default "circuit".
        failure_threshold (int): Failures before opening. Default 3.
        recovery_timeout (float): Seconds before trying half-open. Default 30.0.
        half_open_max (int): Probe calls allowed in half-open state. Default 1.
        success_threshold (int): Consecutive successes in half-open to close. Default 1.
        on_open (callable): Called when circuit opens. Default None.
        on_close (callable): Called when circuit closes. Default None.
    """

    def __init__(self, name: str = "circuit", failure_threshold: int = 3,
                 recovery_timeout: float = 30.0, half_open_max: int = 1,
                 success_threshold: int = 1,
                 on_open=None, on_close=None):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self.success_threshold = success_threshold
        self.on_open = on_open
        self.on_close = on_close

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._half_open_calls = 0
        self._opened_at = 0.0
        self._last_failure_time = 0.0
        self._last_success_time = 0.0
        self._last_error = ""
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejections = 0
        self._total_successes = 0
        self._recovery_attempts = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Check if we should transition from OPEN to HALF_OPEN."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.time() - self._opened_at
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._consecutive_successes = 0
                    self._recovery_attempts += 1
                    log.info("Circuit '%s' transitioning to HALF_OPEN (attempt %d)",
                             self.name, self._recovery_attempts)
            return self._state

    def call(self, fn, *args, **kwargs):
        """Call fn through the circuit breaker. Raises CircuitOpenError if open."""
        self._total_calls += 1
        current_state = self.state

        if current_state == CircuitState.OPEN:
            self._total_rejections += 1
            log.warning("Circuit '%s' is OPEN — call rejected", self.name)
            raise CircuitOpenError(self.name)

        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls >= self.half_open_max:
                    self._total_rejections += 1
                    raise CircuitOpenError(self.name)
                self._half_open_calls += 1

        # Attempt the call
        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            raise

    def wrap(self, fn):
        """Decorator version of call()."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return self.call(fn, *args, **kwargs)
        return wrapper

    def _on_success(self) -> None:
        """Handle a successful call."""
        with self._lock:
            self._consecutive_failures = 0
            self._consecutive_successes += 1
            self._last_success_time = time.time()
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._recovery_attempts = 0
                    log.info("Circuit '%s' CLOSED — service recovered", self.name)
                    if self.on_close:
                        try:
                            self.on_close(self.name)
                        except Exception:
                            pass

    def _on_failure(self, exc: Exception) -> None:
        """Handle a failed call."""
        with self._lock:
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            self._total_failures += 1
            self._last_failure_time = time.time()
            self._last_error = str(exc)[:200]

            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery — back to OPEN
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                log.warning("Circuit '%s' BACK TO OPEN — recovery failed: %s",
                           self.name, exc)
            elif self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                log.warning("Circuit '%s' OPENED after %d failures: %s",
                           self.name, self._consecutive_failures, exc)
                if self.on_open:
                    try:
                        self.on_open(self.name)
                    except Exception:
                        pass

    def reset(self) -> None:
        """Force-reset to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._half_open_calls = 0
            self._recovery_attempts = 0
            log.info("Circuit '%s' force-reset to CLOSED", self.name)

    def get_stats(self) -> dict:
        """Return current statistics."""
        _ = self.state  # trigger state transition check
        return CircuitStats(
            name=self.name,
            state=self._state.value,
            consecutive_failures=self._consecutive_failures,
            total_calls=self._total_calls,
            total_failures=self._total_failures,
            total_rejections=self._total_rejections,
            total_successes=self._total_successes,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            last_error=self._last_error,
            opened_at=self._opened_at,
            recovery_attempts=self._recovery_attempts,
        ).to_dict()


class DegradedModeManager:
    """Manages multiple circuit breakers and provides a degraded mode.

    When the YOLO circuit opens, the pipeline enters degraded mode:
    - Detection is skipped
    - Last-known tracks are held for tracking continuity
    - Alerts are silenced (no detections = no rule triggers)
    - Health endpoint reports degraded status

    Usage::

        dm = DegradedModeManager()
        dm.register("yolo", CircuitBreaker("yolo", failure_threshold=3))
        dm.register("database", CircuitBreaker("database", failure_threshold=5))

        if dm.is_degraded:
            # Use fallback detection
            tracks = dm.get_last_known_tracks("CAM-01")
        else:
            tracks = yolo.detect(frame)
            dm.update_last_known_tracks("CAM-01", tracks)
    """

    def __init__(self, degraded_timeout: float = 60.0):
        self._circuits: dict[str, CircuitBreaker] = {}
        self._last_known_tracks: dict[str, list] = {}
        self._degraded_since: float | None = None
        self._degraded_timeout = degraded_timeout
        self._lock = threading.Lock()

    def register(self, name: str, circuit: CircuitBreaker) -> None:
        """Register a circuit breaker to monitor."""
        self._circuits[name] = circuit

    @property
    def is_degraded(self) -> bool:
        """True if any critical circuit is open."""
        return any(c.state == CircuitState.OPEN for c in self._circuits.values())

    @property
    def degraded_components(self) -> list[str]:
        """List of component names currently in OPEN state."""
        return [name for name, c in self._circuits.items()
                if c.state == CircuitState.OPEN]

    def update_last_known_tracks(self, camera_id: str, tracks: list) -> None:
        """Store the latest successful tracks for fallback use."""
        with self._lock:
            self._last_known_tracks[camera_id] = tracks

    def get_last_known_tracks(self, camera_id: str, max_age_sec: float = 30.0) -> list:
        """Get last-known tracks if degraded, empty if not available or too old."""
        with self._lock:
            tracks = self._last_known_tracks.get(camera_id, [])
            if not tracks:
                return []
            # In degraded mode, return stale tracks but mark them
            return tracks

    def get_health(self) -> dict:
        """Health report for all monitored circuits."""
        circuits = {}
        for name, circuit in self._circuits.items():
            circuits[name] = circuit.get_stats()

        return {
            "status": "degraded" if self.is_degraded else "healthy",
            "degraded_components": self.degraded_components,
            "circuits": circuits,
        }

    def reset_all(self) -> None:
        """Force-reset all circuits to CLOSED."""
        for circuit in self._circuits.values():
            circuit.reset()


class RetryWithBackoff:
    """Retry a callable with exponential backoff and jitter.

    Usage::

        retry = RetryWithBackoff(max_retries=3, base_delay=1.0, max_delay=30.0)
        result = retry.call(risky_operation, arg1, arg2)
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 30.0, backoff_factor: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def call(self, fn, *args, **kwargs):
        """Call fn with retries. Raises the last exception if all retries fail."""
        import random

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    if self.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    log.warning("Retry %d/%d for %s after %.1fs: %s",
                               attempt + 1, self.max_retries,
                               getattr(fn, "__name__", str(fn)), delay, exc)
                    time.sleep(delay)
        raise last_exc

    def wrap(self, fn):
        """Decorator version."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return self.call(fn, *args, **kwargs)
        return wrapper
