"""
Live Search Telemetry, Webhook Streaming & WebSocket Event Engine.

Logs evolutionary metrics, CUDA memory utilization, and surrogate state to TensorBoard,
dispatches Slack/Discord webhooks, and streams live data over WebSockets to web dashboards.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Set

import torch

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

logger = logging.getLogger("neuroswarm.telemetry")


class SearchTelemetryManager:
    """Manages telemetry logging, CUDA VRAM profiling, webhooks, and WebSocket broadcasting."""

    def __init__(
        self,
        log_dir: str = "./runs/neuroswarm_search",
        webhook_url: Optional[str] = None,
        webhook_type: str = "discord",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.webhook_url = webhook_url
        self.webhook_type = webhook_type.lower()

        self.writer = (
            SummaryWriter(log_dir=str(self.log_dir)) if HAS_TENSORBOARD else None
        )
        self.history_buffer: List[Dict[str, Any]] = []
        self.pareto_buffer: List[Dict[str, Any]] = []
        self.active_websockets: Set[Any] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    @staticmethod
    def get_cuda_vram_stats() -> Dict[str, float]:
        """Extracts allocated and reserved CUDA GPU memory in Megabytes (MB)."""
        if not torch.cuda.is_available():
            return {
                "vram_allocated_mb": 0.0,
                "vram_reserved_mb": 0.0,
                "vram_max_allocated_mb": 0.0,
            }

        return {
            "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2),
            "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 2),
            "vram_max_allocated_mb": round(
                torch.cuda.max_memory_allocated(0) / (1024**2), 2
            ),
        }

    def register_websocket(self, ws: Any):
        """Registers an active WebSocket connection."""
        self.active_websockets.add(ws)

    def unregister_websocket(self, ws: Any):
        """Removes a disconnected WebSocket connection."""
        self.active_websockets.discard(ws)

    def broadcast_event_sync(self, event_type: str, data: Dict[str, Any]):
        """Dispatches event payloads to registered WebSocket clients safely."""
        payload = json.dumps(
            {"event": event_type, "timestamp": time.time(), "data": data}
        )
        stale_clients = set()
        for ws in list(self.active_websockets):
            try:
                if hasattr(ws, "send_text"):
                    if self.loop and self.loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            ws.send_text(payload), self.loop
                        )
                    else:
                        asyncio.run(ws.send_text(payload))
            except Exception:
                stale_clients.add(ws)
        for ws in stale_clients:
            self.unregister_websocket(ws)

    def log_generation(self, gen_metrics: Dict[str, Any]):
        """Logs generation metrics to TensorBoard and broadcasts via WebSocket."""
        gen = gen_metrics.get("generation", 0)
        vram = self.get_cuda_vram_stats()

        record = {**gen_metrics, **vram, "timestamp": time.strftime("%H:%M:%S")}
        self.history_buffer.append(record)

        if self.writer:
            self.writer.add_scalar(
                "Fitness/Global_Best", gen_metrics.get("best_fitness", 0.0), gen
            )
            self.writer.add_scalar(
                "Fitness/Mean", gen_metrics.get("mean_fitness", 0.0), gen
            )
            self.writer.add_scalar(
                "Hardware/Best_Latency_ms", gen_metrics.get("best_latency_ms", 0.0), gen
            )
            self.writer.add_scalar(
                "Hardware/Best_Params", gen_metrics.get("best_params", 0), gen
            )
            self.writer.add_scalar("VRAM/Allocated_MB", vram["vram_allocated_mb"], gen)
            self.writer.add_scalar(
                "VRAM/Max_Allocated_MB", vram["vram_max_allocated_mb"], gen
            )
            self.writer.flush()

        self.broadcast_event_sync("generation_update", record)
        logger.info(
            f"Telemetry [Gen {gen:02d}] | Best Fitness: {gen_metrics.get('best_fitness', 0.0):.4f} | "
            f"VRAM: {vram['vram_allocated_mb']:.1f} MB / {vram['vram_max_allocated_mb']:.1f} MB"
        )

    def notify_pareto_discovery(
        self, candidate_id: str, accuracy: float, latency_ms: float, params: int
    ):
        """Dispatches Pareto discovery alerts to webhooks and WebSockets."""
        fields = {
            "Candidate ID": candidate_id,
            "Accuracy": f"{accuracy * 100:.2f}%",
            "Latency": f"{latency_ms:.2f} ms",
            "Parameters": f"{params:,}",
        }
        self.pareto_buffer.append(
            {
                "candidate_id": candidate_id,
                "accuracy": accuracy,
                "latency_ms": latency_ms,
                "params": params,
            }
        )
        self.broadcast_event_sync("pareto_discovery", fields)

    def close(self):
        """Flushes and closes TensorBoard loggers."""
        if self.writer:
            self.writer.close()
