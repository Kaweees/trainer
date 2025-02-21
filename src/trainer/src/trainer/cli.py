import typer

from trainer.mnist_training import train

app = typer.Typer()


@app.command()
def run():
    train()


def entrypoint():
    app()
