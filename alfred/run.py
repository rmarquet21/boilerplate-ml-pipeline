import alfred


@alfred.command("run:pipeline", help="Run the ML Pipeline.")
def run_pipeline():
    python = alfred.sh("python", "python should be present")
    alfred.run(python, ["training_pipeline/run.py"])


@alfred.command("run:server", help="Run the MLflow Server.")
def run_server():
    mlflow = alfred.sh("mlflow")
    args = ["ui"]
    alfred.run(mlflow, args)
