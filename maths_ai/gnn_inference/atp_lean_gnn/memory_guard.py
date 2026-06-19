from __future__ import annotations

import argparse
import os
import resource
from dataclasses import dataclass

import torch


try:
    import psutil
except ImportError:  # pragma: no cover - psutil is an optional runtime helper.
    psutil = None


BYTES_PER_GB = 1024**3


class MemoryLimitExceeded(RuntimeError):
    """Raised when a training process crosses a configured memory limit."""


@dataclass(frozen=True)
class MemorySnapshot:
    process_gb: float | None
    system_percent: float | None
    cuda_allocated_gb: float | None
    cuda_reserved_gb: float | None
    cuda_total_gb: float | None

    @property
    def cuda_allocated_percent(self) -> float | None:
        if self.cuda_allocated_gb is None or not self.cuda_total_gb:
            return None
        return 100.0 * self.cuda_allocated_gb / self.cuda_total_gb

    @property
    def cuda_reserved_percent(self) -> float | None:
        if self.cuda_reserved_gb is None or not self.cuda_total_gb:
            return None
        return 100.0 * self.cuda_reserved_gb / self.cuda_total_gb

    def format(self) -> str:
        parts: list[str] = []
        if self.process_gb is not None:
            parts.append(f"process={self.process_gb:.2f}GB")
        if self.system_percent is not None:
            parts.append(f"system={self.system_percent:.1f}%")
        if self.cuda_allocated_gb is not None:
            percent = self.cuda_allocated_percent
            suffix = "" if percent is None else f" ({percent:.1f}%)"
            parts.append(f"cuda_allocated={self.cuda_allocated_gb:.2f}GB{suffix}")
        if self.cuda_reserved_gb is not None:
            percent = self.cuda_reserved_percent
            suffix = "" if percent is None else f" ({percent:.1f}%)"
            parts.append(f"cuda_reserved={self.cuda_reserved_gb:.2f}GB{suffix}")
        return ", ".join(parts) if parts else "memory metrics unavailable"


@dataclass(frozen=True)
class MemoryGuardConfig:
    max_process_memory_gb: float | None = None
    max_system_memory_percent: float | None = 90.0
    max_cuda_allocated_gb: float | None = None
    max_cuda_allocated_percent: float | None = None
    max_cuda_reserved_gb: float | None = None
    max_cuda_reserved_percent: float | None = 90.0
    check_every_batches: int = 1
    enabled: bool = True

    def normalized(self) -> "MemoryGuardConfig":
        return MemoryGuardConfig(
            max_process_memory_gb=_positive_or_none(self.max_process_memory_gb),
            max_system_memory_percent=_percent_or_none(self.max_system_memory_percent),
            max_cuda_allocated_gb=_positive_or_none(self.max_cuda_allocated_gb),
            max_cuda_allocated_percent=_percent_or_none(self.max_cuda_allocated_percent),
            max_cuda_reserved_gb=_positive_or_none(self.max_cuda_reserved_gb),
            max_cuda_reserved_percent=_percent_or_none(self.max_cuda_reserved_percent),
            check_every_batches=max(int(self.check_every_batches), 1),
            enabled=bool(self.enabled),
        )

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "enabled": normalized.enabled,
            "max_process_memory_gb": normalized.max_process_memory_gb,
            "max_system_memory_percent": normalized.max_system_memory_percent,
            "max_cuda_allocated_gb": normalized.max_cuda_allocated_gb,
            "max_cuda_allocated_percent": normalized.max_cuda_allocated_percent,
            "max_cuda_reserved_gb": normalized.max_cuda_reserved_gb,
            "max_cuda_reserved_percent": normalized.max_cuda_reserved_percent,
            "check_every_batches": normalized.check_every_batches,
        }


class MemoryGuard:
    def __init__(self, config: MemoryGuardConfig, *, device: torch.device) -> None:
        self.config = config.normalized()
        self.device = device

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def describe(self) -> str:
        if not self.enabled:
            return "disabled"
        limits: list[str] = [f"check_every_batches={self.config.check_every_batches}"]
        if self.config.max_process_memory_gb is not None:
            limits.append(f"process<={self.config.max_process_memory_gb:.2f}GB")
        if self.config.max_system_memory_percent is not None:
            limits.append(f"system<={self.config.max_system_memory_percent:.1f}%")
        if self.config.max_cuda_allocated_gb is not None:
            limits.append(f"cuda_allocated<={self.config.max_cuda_allocated_gb:.2f}GB")
        if self.config.max_cuda_allocated_percent is not None:
            limits.append(f"cuda_allocated<={self.config.max_cuda_allocated_percent:.1f}%")
        if self.config.max_cuda_reserved_gb is not None:
            limits.append(f"cuda_reserved<={self.config.max_cuda_reserved_gb:.2f}GB")
        if self.config.max_cuda_reserved_percent is not None:
            limits.append(f"cuda_reserved<={self.config.max_cuda_reserved_percent:.1f}%")
        return ", ".join(limits)

    def check_batch(self, label: str, batch_index: int) -> MemorySnapshot | None:
        if batch_index == 1 or batch_index % self.config.check_every_batches == 0:
            return self.check(label)
        return None

    def check(self, label: str) -> MemorySnapshot | None:
        if not self.enabled:
            return None
        snapshot = snapshot_memory(self.device)
        violations = _violations(snapshot, self.config)
        if violations:
            details = "; ".join(violations)
            raise MemoryLimitExceeded(
                f"Memory guard stopped training at {label}: {details}. "
                f"Current memory: {snapshot.format()}"
            )
        return snapshot


def add_memory_guard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--disable-memory-guard",
        action="store_true",
        help="Disable training memory checks",
    )
    parser.add_argument(
        "--memory-check-every-batches",
        type=int,
        default=1,
        help="How often to check memory during training/evaluation batches",
    )
    parser.add_argument(
        "--max-process-memory-gb",
        type=float,
        default=None,
        help="Stop if this process RSS exceeds the given GB value; <=0 disables",
    )
    parser.add_argument(
        "--max-system-memory-percent",
        type=float,
        default=90.0,
        help="Stop if total system RAM usage exceeds this percent; <=0 disables",
    )
    parser.add_argument(
        "--max-cuda-allocated-gb",
        type=float,
        default=None,
        help="Stop if CUDA allocated memory exceeds the given GB value; <=0 disables",
    )
    parser.add_argument(
        "--max-cuda-allocated-percent",
        type=float,
        default=None,
        help="Stop if CUDA allocated memory exceeds this percent of device memory; <=0 disables",
    )
    parser.add_argument(
        "--max-cuda-reserved-gb",
        type=float,
        default=None,
        help="Stop if CUDA reserved memory exceeds the given GB value; <=0 disables",
    )
    parser.add_argument(
        "--max-cuda-reserved-percent",
        type=float,
        default=90.0,
        help="Stop if CUDA reserved memory exceeds this percent of device memory; <=0 disables",
    )


def memory_guard_from_args(args: argparse.Namespace, *, device: torch.device) -> MemoryGuard:
    return MemoryGuard(
        MemoryGuardConfig(
            enabled=not bool(args.disable_memory_guard),
            check_every_batches=int(args.memory_check_every_batches),
            max_process_memory_gb=args.max_process_memory_gb,
            max_system_memory_percent=args.max_system_memory_percent,
            max_cuda_allocated_gb=args.max_cuda_allocated_gb,
            max_cuda_allocated_percent=args.max_cuda_allocated_percent,
            max_cuda_reserved_gb=args.max_cuda_reserved_gb,
            max_cuda_reserved_percent=args.max_cuda_reserved_percent,
        ),
        device=device,
    )


def snapshot_memory(device: torch.device) -> MemorySnapshot:
    process_gb = _process_memory_gb()
    system_percent = _system_memory_percent()
    cuda_allocated_gb = None
    cuda_reserved_gb = None
    cuda_total_gb = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_device = device if device.index is not None else torch.device("cuda", torch.cuda.current_device())
        cuda_allocated_gb = torch.cuda.memory_allocated(cuda_device) / BYTES_PER_GB
        cuda_reserved_gb = torch.cuda.memory_reserved(cuda_device) / BYTES_PER_GB
        cuda_total_gb = torch.cuda.get_device_properties(cuda_device).total_memory / BYTES_PER_GB
    return MemorySnapshot(
        process_gb=process_gb,
        system_percent=system_percent,
        cuda_allocated_gb=cuda_allocated_gb,
        cuda_reserved_gb=cuda_reserved_gb,
        cuda_total_gb=cuda_total_gb,
    )


def _violations(snapshot: MemorySnapshot, config: MemoryGuardConfig) -> list[str]:
    violations: list[str] = []
    if (
        config.max_process_memory_gb is not None
        and snapshot.process_gb is not None
        and snapshot.process_gb > config.max_process_memory_gb
    ):
        violations.append(
            f"process {snapshot.process_gb:.2f}GB > {config.max_process_memory_gb:.2f}GB"
        )
    if (
        config.max_system_memory_percent is not None
        and snapshot.system_percent is not None
        and snapshot.system_percent > config.max_system_memory_percent
    ):
        violations.append(
            f"system {snapshot.system_percent:.1f}% > {config.max_system_memory_percent:.1f}%"
        )
    if (
        config.max_cuda_allocated_gb is not None
        and snapshot.cuda_allocated_gb is not None
        and snapshot.cuda_allocated_gb > config.max_cuda_allocated_gb
    ):
        violations.append(
            f"cuda allocated {snapshot.cuda_allocated_gb:.2f}GB > {config.max_cuda_allocated_gb:.2f}GB"
        )
    if (
        config.max_cuda_allocated_percent is not None
        and snapshot.cuda_allocated_percent is not None
        and snapshot.cuda_allocated_percent > config.max_cuda_allocated_percent
    ):
        violations.append(
            f"cuda allocated {snapshot.cuda_allocated_percent:.1f}% > {config.max_cuda_allocated_percent:.1f}%"
        )
    if (
        config.max_cuda_reserved_gb is not None
        and snapshot.cuda_reserved_gb is not None
        and snapshot.cuda_reserved_gb > config.max_cuda_reserved_gb
    ):
        violations.append(
            f"cuda reserved {snapshot.cuda_reserved_gb:.2f}GB > {config.max_cuda_reserved_gb:.2f}GB"
        )
    if (
        config.max_cuda_reserved_percent is not None
        and snapshot.cuda_reserved_percent is not None
        and snapshot.cuda_reserved_percent > config.max_cuda_reserved_percent
    ):
        violations.append(
            f"cuda reserved {snapshot.cuda_reserved_percent:.1f}% > {config.max_cuda_reserved_percent:.1f}%"
        )
    return violations


def _positive_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def _percent_or_none(value: float | None) -> float | None:
    value = _positive_or_none(value)
    if value is None:
        return None
    return min(value, 100.0)


def _process_memory_gb() -> float | None:
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / BYTES_PER_GB

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    # Linux reports KiB, macOS reports bytes.
    usage_bytes = usage if usage > 10_000_000_000 else usage * 1024
    return usage_bytes / BYTES_PER_GB


def _system_memory_percent() -> float | None:
    if psutil is not None:
        return float(psutil.virtual_memory().percent)
    return None
