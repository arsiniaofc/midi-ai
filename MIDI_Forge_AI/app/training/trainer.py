"""
MIDI Forge AI - Training Engine Module
Executes real local PyTorch Transformer training in a background thread.
Tracks loss, learning rate, checkpoints, ETA, and hardware metrics.
Supports pause, resume, and graceful abort without corruption.
"""

import os
import time
import math
import threading
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

try:
    import torch
    import torch.optim as optim
except ImportError:
    torch = None
    optim = None

from .model import MusicGPT, GPTConfig
from .dataset import MusicDataset


@dataclass
class TrainProgress:
    status: str = "idle"  # idle, running, paused, stopped, completed, error
    model_name: str = ""
    current_epoch: int = 0
    total_epochs: int = 10
    current_step: int = 0
    total_steps: int = 0
    epoch_progress_pct: float = 0.0
    overall_progress_pct: float = 0.0
    current_train_loss: float = 0.0
    current_val_loss: float = 0.0
    best_val_loss: float = float("inf")
    learning_rate: float = 0.0005
    elapsed_sec: float = 0.0
    remaining_sec: float = 0.0
    device: str = "cpu"
    gpu_name: Optional[str] = None
    latest_checkpoint: Optional[str] = None
    error_message: Optional[str] = None
    loss_history: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["loss_history"] is None:
            d["loss_history"] = []
        return d


class LocalMusicTrainer:
    """Thread-safe training supervisor with pause, resume, and checkpoint recovery."""

    def __init__(self, checkpoints_dir: str):
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(checkpoints_dir, exist_ok=True)

        self.progress = TrainProgress(loss_history=[])
        self.lock = threading.Lock()

        self._stop_requested = False
        self._pause_requested = False
        self._thread: Optional[threading.Thread] = None

        self.model: Optional[MusicGPT] = None
        self.dataset: Optional[MusicDataset] = None
        self.optimizer = None
        self.device = "cpu"

    def get_progress(self) -> Dict[str, Any]:
        with self.lock:
            return self.progress.to_dict()

    def pause(self):
        with self.lock:
            self._pause_requested = True
            if self.progress.status == "running":
                self.progress.status = "paused"

    def resume(self):
        with self.lock:
            self._pause_requested = False
            if self.progress.status == "paused":
                self.progress.status = "running"

    def stop(self):
        with self.lock:
            self._stop_requested = True
            self.progress.status = "stopped"

    def start_training(
        self,
        model: MusicGPT,
        dataset: MusicDataset,
        model_name: str,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 5e-4,
        warmup_steps: int = 50,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        checkpoint_every_epochs: int = 1,
        resume_checkpoint_path: Optional[str] = None,
    ):
        """Starts asynchronous training loop in a dedicated background thread."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("O treinamento já está em execução.")

        self.model = model
        self.dataset = dataset
        self._stop_requested = False
        self._pause_requested = False

        self.progress = TrainProgress(
            status="running",
            model_name=model_name,
            total_epochs=epochs,
            learning_rate=learning_rate,
            loss_history=[],
        )

        self._thread = threading.Thread(
            target=self._run_training_loop,
            kwargs=dict(
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                checkpoint_freq=checkpoint_every_epochs,
                resume_path=resume_checkpoint_path,
            ),
            daemon=True,
        )
        self._thread.start()

    def _run_training_loop(
        self,
        epochs: int,
        batch_size: int,
        lr: float,
        warmup_steps: int,
        weight_decay: float,
        grad_clip: float,
        checkpoint_freq: int,
        resume_path: Optional[str],
    ):
        if torch is None:
            with self.lock:
                self.progress.status = "error"
                self.progress.error_message = "PyTorch não está instalado no ambiente."
            return

        # Select hardware: auto CUDA or CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        with self.lock:
            self.progress.device = device
            if device == "cuda":
                self.progress.gpu_name = torch.cuda.get_device_name(0)

        self.model.to(device)
        self.model.train()

        # Optimizer: AdamW
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
        self.optimizer = optimizer

        start_epoch = 0
        global_step = 0

        # Load checkpoint if resuming
        if resume_path and os.path.exists(resume_path):
            try:
                ckpt = torch.load(resume_path, map_location=device)
                self.model.load_state_dict(ckpt["model_state"])
                if "optimizer_state" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                start_epoch = ckpt.get("epoch", 0) + 1
                global_step = ckpt.get("step", 0)
            except Exception as e:
                with self.lock:
                    self.progress.status = "error"
                    self.progress.error_message = f"Falha ao carregar checkpoint de retomada: {str(e)}"
                return

        steps_per_epoch = max(1, len(self.dataset.train_samples) // batch_size)
        total_steps = epochs * steps_per_epoch
        with self.lock:
            self.progress.total_steps = total_steps
            self.progress.current_epoch = start_epoch

        start_time = time.time()

        try:
            for epoch in range(start_epoch, epochs):
                with self.lock:
                    self.progress.current_epoch = epoch

                for step_in_epoch in range(steps_per_epoch):
                    # Check for stop request
                    if self._stop_requested:
                        with self.lock:
                            self.progress.status = "stopped"
                        self._save_checkpoint_file(epoch, global_step, optimizer, prefix="aborted_")
                        return

                    # Handle pause state
                    while self._pause_requested:
                        time.sleep(0.5)
                        if self._stop_requested:
                            with self.lock:
                                self.progress.status = "stopped"
                            return

                    # Learning rate schedule with warmup & cosine decay
                    global_step += 1
                    if global_step < warmup_steps:
                        curr_lr = lr * (global_step / max(1, warmup_steps))
                    else:
                        decay_ratio = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                        curr_lr = lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_ratio)))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr

                    # Fetch batch and execute training step
                    x, y = self.dataset.get_batch("train", batch_size=batch_size, device=device)
                    optimizer.zero_grad()

                    logits, loss = self.model(x, targets=y)
                    loss_val = loss.item()

                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer.step()

                    # Periodic validation evaluation
                    val_loss_val = loss_val
                    if step_in_epoch % max(1, steps_per_epoch // 4) == 0:
                        self.model.eval()
                        with torch.no_grad():
                            x_val, y_val = self.dataset.get_batch("val", batch_size=batch_size, device=device)
                            _, v_loss = self.model(x_val, targets=y_val)
                            val_loss_val = v_loss.item()
                        self.model.train()

                    # Update progress metrics
                    now = time.time()
                    elapsed = now - start_time
                    steps_done = max(1, global_step)
                    time_per_step = elapsed / steps_done
                    remaining = max(0.0, (total_steps - steps_done) * time_per_step)

                    with self.lock:
                        self.progress.current_step = global_step
                        self.progress.current_train_loss = round(loss_val, 4)
                        self.progress.current_val_loss = round(val_loss_val, 4)
                        self.progress.learning_rate = round(curr_lr, 6)
                        self.progress.elapsed_sec = round(elapsed, 1)
                        self.progress.remaining_sec = round(remaining, 1)
                        self.progress.epoch_progress_pct = round((step_in_epoch + 1) / steps_per_epoch * 100.0, 1)
                        self.progress.overall_progress_pct = round(min(100.0, global_step / total_steps * 100.0), 1)

                        if len(self.progress.loss_history) < 200 or global_step % 5 == 0:
                            self.progress.loss_history.append({
                                "step": global_step,
                                "epoch": epoch + 1,
                                "train_loss": round(loss_val, 4),
                                "val_loss": round(val_loss_val, 4),
                            })
                            if len(self.progress.loss_history) > 300:
                                self.progress.loss_history = self.progress.loss_history[-300:]

                # Save checkpoint at epoch boundary
                if (epoch + 1) % checkpoint_freq == 0 or epoch == epochs - 1:
                    ckpt_path = self._save_checkpoint_file(epoch, global_step, optimizer)
                    with self.lock:
                        self.progress.latest_checkpoint = ckpt_path

            with self.lock:
                self.progress.status = "completed"
                self.progress.overall_progress_pct = 100.0

        except Exception as e:
            with self.lock:
                self.progress.status = "error"
                self.progress.error_message = f"Erro no treinamento: {str(e)}"

    def _save_checkpoint_file(
        self, epoch: int, step: int, optimizer: Any, prefix: str = ""
    ) -> str:
        filename = f"{prefix}{self.progress.model_name}_epoch_{epoch + 1}_step_{step}.pt"
        ckpt_path = os.path.join(self.checkpoints_dir, filename)

        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "model_config": self.model.config.to_dict(),
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "train_loss": self.progress.current_train_loss,
            "val_loss": self.progress.current_val_loss,
            "timestamp": time.time(),
        }
        torch.save(checkpoint_data, ckpt_path)
        return ckpt_path
