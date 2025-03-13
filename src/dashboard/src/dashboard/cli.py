import typer
import uvicorn
from utils import get_token

app = typer.Typer()


@app.command()
def run():
    """Start the FastAPI server"""
    uvicorn.run(
        "dashboard.api:api",
        host=get_token("DASHBOARD_HOST"),
        port=int(get_token("DASHBOARD_PORT")),
        reload=True,
    )


def entrypoint():
    app()
