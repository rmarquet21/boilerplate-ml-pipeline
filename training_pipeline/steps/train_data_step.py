import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from .base_step import BaseStep
from tqdm import tqdm
import logging
import mlflow


class TrainDataStep(BaseStep):
    def execute(self, data, dependencies):
        logging.info("Training BERT...")
        mlflow_client: mlflow = dependencies["mlflow_client"]

        batch_size = data.get("batch_size", 8)
        num_epochs = data.get("num_epochs", 10)
        learning_rate = data.get("learning_rate", 5e-5)
        validate_every = data.get("validate_every", 1)
        early_stopping_patience = data.get("early_stopping_patience", 3)
        gradient_clip_val = data.get("gradient_clip_val", None)
        device = data.get("device", None)

        mlflow_client.log_param("batch_size", batch_size)
        mlflow_client.log_param("num_epochs", num_epochs)
        mlflow_client.log_param("learning_rate", learning_rate)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        train_loader = DataLoader(
            data["train_encodings"], shuffle=True, batch_size=batch_size
        )
        val_loader = DataLoader(data["val_encodings"], batch_size=batch_size)
        test_loader = DataLoader(data["test_encodings"], batch_size=batch_size)

        model = dependencies["model"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.to(device)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * num_epochs,
        )

        criterion = torch.nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        no_improve_counter = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
                optimizer.zero_grad()

                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).long()

                outputs = model(inputs, attention_mask=attention_mask).logits
                loss = criterion(outputs, labels)

                loss.backward()
                if gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_val
                    )
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}"
            )
            mlflow_client.log_metric("train_loss", avg_train_loss, step=epoch + 1)

            if epoch % validate_every == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device).long()

                        outputs = model(inputs, attention_mask=attention_mask).logits
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                logging.info(
                    f"Validation Loss after Epoch {epoch + 1}/{num_epochs}: {avg_val_loss:.4f}"
                )
                mlflow_client.log_metric("val_loss", avg_val_loss, step=epoch + 1)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                    if no_improve_counter >= early_stopping_patience:
                        logging.info(
                            "Early stopping due to no improvement on validation loss."
                        )
                        break

        # Testing loop
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).long()

                outputs = model(inputs, attention_mask=attention_mask).logits
                _, predicted = torch.max(
                    outputs, 1
                )  # Get index of max value along axis 1

                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
        mlflow_client.log_metric("test_accuracy", accuracy)

        # Save model
        mlflow_client.pytorch.log_model(model, "models")
        data["model"] = model
        data["score"] = accuracy * 100
        return data
