import mlflow


class PipelineContext:
    def __init__(self):
        self.data = {}
        self.dependencies = {}
        self.mlflow_client = mlflow
