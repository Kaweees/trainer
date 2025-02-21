import asyncio
from typing import Any

import torch

from .cli import broadcast_metrics


class DashboardCallback:
    def __init__(self):
        self.iteration = 0
        self.metrics: dict[str, Any] = {
            "loss": 0.0,
            "perplexity": 0.0,
            "accuracy": 0.0,
            "learning_rate": 0.0,
            "gradient_norm": 0.0,
            "bits_memorized": 0.0,
            "bits_per_second": 0.0,
            "gpu_utilization": [],
            "skills": {
                "Translation": 0,
                "Summarization": 0,
                "Reasoning": 0,
                "Coding": 0,
                "Comprehension": 0,
            },
        }

    def on_train_batch_end(self, trainer, model, outputs):
        self.iteration += 1

        # Collect metrics
        self.metrics.update(
            {
                "loss": outputs["loss"].item(),
                "perplexity": torch.exp(outputs["loss"]).item(),
                "accuracy": outputs.get("accuracy", 0.0),
                "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
                "gradient_norm": self._get_gradient_norm(model),
                "bits_memorized": self.iteration * 0.5,  # Example calculation
                "gpu_utilization": self._get_gpu_utilization(),
            }
        )

        # Broadcast metrics
        asyncio.create_task(broadcast_metrics(self.metrics))

    def _get_gradient_norm(self, model) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return torch.sqrt(torch.tensor(total_norm)).item()

    def _get_gpu_utilization(self) -> list:
        if not torch.cuda.is_available():
            return [0.0] * 1024

        # Get actual GPU utilization (example using nvidia-smi)
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            utils = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utils.append(util.gpu / 100.0)
            return utils + [0.0] * (1024 - len(utils))
        except (ImportError, pynvml.NVMLError) as e:
            print(f"Failed to get GPU utilization: {e}")
            return [0.0] * 1024
