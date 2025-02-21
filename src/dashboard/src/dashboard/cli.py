from pathlib import Path

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

# Configure templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@api.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Renders the dashboard HTML template"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@api.get("/number")
async def get_number():
    """Endpoint that returns the result of return_two()"""
    return {"number": return_two()}


@app.command()
def run():
    """Start the FastAPI server"""
    uvicorn.run("dashboard.cli:api", host="0.0.0.0", port=8080, reload=True)
