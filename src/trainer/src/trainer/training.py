import requests
import torch
from utils import get_token


def send_metrics_to_dashboard(trainer_state):
    """
    Send current training metrics to the dashboard

    Args:
        trainer_state: Current training state containing metrics
    """
    # Convert loss to tensor if it's a float
    loss_tensor = (
        torch.tensor(trainer_state.loss)
        if isinstance(trainer_state.loss, float)
        else trainer_state.loss
    )

    metrics = {
        "iteration": trainer_state.iteration,
        "loss": float(trainer_state.loss),
        "perplexity": float(
            torch.exp(loss_tensor)
        ),  # Convert loss to tensor before exp
        "accuracy": float(trainer_state.accuracy),
        "learning_rate": float(trainer_state.optimizer.param_groups[0]["lr"]),
        "gradient_norm": float(
            torch.nn.utils.clip_grad_norm_(
                trainer_state.model.parameters(), float("inf")
            )
        ),
        "bits_memorized": trainer_state.bits_memorized,
        "bits_per_second": trainer_state.bits_per_second,
        "gpu_utilization": trainer_state.gpu_utilization,
        "skills": trainer_state.skills,
    }

    try:
        requests.post(
            f"http://{get_token('DASHBOARD_HOST')}:{get_token('DASHBOARD_PORT')}/update_metrics",
            json=metrics,
        )
    except Exception as e:
        print(f"Failed to update dashboard: {e}")


# def train():
#     """Start the training process"""
#     print("Starting training...")
#     # Add your training logic here
#     # In your training loop:
#     for epoch in range(num_epochs):
#         for batch in dataloader:
#             # ... training step ...

#             # Update dashboard every N steps
#             if step % update_frequency == 0:
#                 send_metrics_to_dashboard(trainer_state)


# def evaluate():
#     """Evaluate a trained model"""
#     print("Starting evaluation...")
#     # Add your evaluation logic here
