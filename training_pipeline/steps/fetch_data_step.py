from .base_step import BaseStep
from datasets import load_dataset
import logging
import mlflow


class FetchDataStep(BaseStep):
    def execute(self, data, dependencies):
        mlflow_client: mlflow = dependencies["mlflow_client"]

        logging.info("Fetching Rotten Tomatoes data...")
        dataset = load_dataset("rotten_tomatoes")

        mlflow_client.log_param("dataset", "rotten_tomatoes")
        mlflow_client.log_param("train_size", len(dataset["train"]))
        mlflow_client.log_param("test_size", len(dataset["test"]))
        mlflow_client.log_param("validation_size", len(dataset["validation"]))

        # Store them in the pipeline data dictionary
        data["train"] = dataset["train"]
        data["test"] = dataset["test"]
        data["val"] = dataset["validation"]

        return data
