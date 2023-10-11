from .base_step import BaseStep
import logging


class PrepareDataStep(BaseStep):
    def execute(self, data, dependencies):
        logging.info("Preparing data for BERT...")
        tokenizer = dependencies["tokenizer"]

        # Encode texts and include labels
        def encode_and_add_labels(example):
            encoding = tokenizer(example["text"], truncation=True, padding="max_length")
            encoding["labels"] = example["label"]
            return encoding

        train_encodings = data["train"].map(encode_and_add_labels, batched=True)
        test_encodings = data["test"].map(encode_and_add_labels, batched=True)
        val_encodings = data["val"].map(encode_and_add_labels, batched=True)

        # Convert the datasets to PyTorch format
        train_encodings.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        test_encodings.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        val_encodings.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        data["train_encodings"] = train_encodings
        data["test_encodings"] = test_encodings
        data["val_encodings"] = val_encodings
        return data
