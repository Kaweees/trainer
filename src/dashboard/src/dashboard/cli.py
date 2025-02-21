import typer
import uvicorn
from fastapi import FastAPI

from core import return_two

app = typer.Typer()
api = FastAPI(
    title="Dashboard API",
    description="API for the dashboard application",
    version="1.0.0",
)


@api.get("/")
async def root():
    """Root endpoint that returns a welcome message"""
    return {"message": "Welcome to the Dashboard API"}


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
