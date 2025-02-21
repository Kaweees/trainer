import typer

app = typer.Typer()


@app.command()
def run():
    print("Wassup")


def entrypoint():
    app()
