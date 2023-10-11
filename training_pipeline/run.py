import logging
import mlflow
from transformers import BertForSequenceClassification, BertTokenizer
from steps.fetch_data_step import FetchDataStep
from steps.clean_data_step import CleanDataStep
from steps.prepare_data_step import PrepareDataStep
from steps.train_data_step import TrainDataStep
from pipeline_context import PipelineContext

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    # Set up MLflow
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )  # default local URI, change if you have a different setup
    mlflow.set_experiment("Rotten Tomatoes Sentiment Analysis")

    with mlflow.start_run():
        # Create the pipeline context
        context = PipelineContext()
        context.dependencies["model"] = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased"
        )
        context.dependencies["tokenizer"] = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        context.dependencies["mlflow_client"] = mlflow

        steps = [FetchDataStep(), CleanDataStep(), PrepareDataStep(), TrainDataStep()]

        for step in steps:
            try:
                context.data = step.execute(context.data, context.dependencies)
            except Exception as e:
                logging.error(f"Error executing step {step.__class__.__name__}: {e}")
                return

        logging.info(
            f"Model Training Complete with Accuracy: {context.data['score']:.2f}"
        )


if __name__ == "__main__":
    main()
