import os

import typer
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from core import return_two

app = typer.Typer()
api = FastAPI(
    title="Dashboard API",
    description="API for the dashboard application",
    version="1.0.0",
)

# Set up templates directory
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

# Training metrics storage
training_metrics = {
    "iteration": 0,
    "loss": 0.0,
    "perplexity": 0.0,
    "accuracy": 0.0,
    "learning_rate": 0.0,
    "gradient_norm": 0.0,
    "bits_memorized": 0.0,
    "bits_per_second": 0.0,
    "gpu_utilization": [0.0] * 1024,  # For 1024 GPUs
    "skills": {
        "Translation": 20,
        "Summarization": 15,
        "Reasoning": 10,
        "Coding": 5,
        "Comprehension": 25,
    },
}


@api.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the dashboard HTML"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@api.post("/update_metrics")
async def update_metrics(metrics: dict):
    """Receive updated metrics from PyTorch training"""
    # Update our stored metrics
    training_metrics.update(metrics)
    return {"status": "success"}


@api.get("/metrics")
async def get_metrics():
    """Return current training metrics"""
    return training_metrics


@api.get("/number")
async def get_number():
    """Endpoint that returns the result of return_two()"""
    return {"number": return_two()}


@app.command()
def run():
    """Start the FastAPI server"""
    uvicorn.run("dashboard.cli:api", host="0.0.0.0", port=8080, reload=True)


def entrypoint():
    app()
