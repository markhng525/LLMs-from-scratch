import time
from itertools import islice
from pathlib import Path

import pandas as pd
import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from ch04.practice.gpt_model import GPTModelV2

PROJ_ROOT = "/workspace/ch06"


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def get_train_test_dataset_split() -> tuple[Dataset, Dataset, Dataset]:
    subproj_dir = "01_main-chapter-code"
    train_dataset = SpamDataset(
        csv_file=Path(PROJ_ROOT) / subproj_dir / "train.csv",
        tokenizer=tokenizer,
        max_length=None,
    )
    return (
        train_dataset,
        SpamDataset(
            csv_file=Path(PROJ_ROOT) / subproj_dir / "validation.csv",
            tokenizer=tokenizer,
            max_length=train_dataset.max_length,
        ),
        SpamDataset(
            csv_file=Path(PROJ_ROOT) / subproj_dir / "test.csv",
            tokenizer=tokenizer,
            max_length=train_dataset.max_length,
        ),
    )


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModelV2,
    device: str,
) -> torch.Tensor:
    """Calculate the cross entropy loss for classification scoring the final token in the sequence"""
    input_batch = input_batch.to(dtype=torch.long, device=device)
    target_batch = target_batch.to(dtype=torch.long, device=device)
    num_classes = model.lm_head.weight.shape[0]
    logits = model(input_batch)
    logits_flat = logits[:, -1, :].reshape(-1, num_classes)
    target_flat = target_batch.reshape(-1)
    loss = F.cross_entropy(logits_flat, target_flat)
    return loss


def calc_accuracy_loader(
    data_loader: DataLoader,
    model: GPTModelV2,
    device: str,
    num_batches: int | None = None,
) -> float:
    predicted_labels = []
    if num_batches is None:
        batches = data_loader
    else:
        batches = islice(data_loader, num_batches)

    model.eval()
    num_samples = 0
    correct = 0
    for input_batch, target_batch in batches:
        input_batch.to(device)
        target_batch.to(device)
        with torch.no_grad():
            logits: torch.Tensor = model(input_batch)
        predicted_labels = logits[:, -1, :].argmax(dim=-1)
        correct += (predicted_labels == target_batch).sum().item()
        num_samples += predicted_labels.shape[-1]
    model.training = True

    return correct / num_samples


def calc_loss_loader(
    data_loader: DataLoader,
    model: GPTModelV2,
    device: str,
    num_batches: int | None = None,
) -> float:
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        batches = data_loader
    else:
        if num_batches > len(data_loader):
            print(
                "Warning: num_batches > total batches in data_loader."
                "Truncating to len(data_loader)"
            )
        batches = islice(data_loader, num_batches)

    count = 0
    for input_batch, target_batch in batches:
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
        count += 1

    return total_loss / count if count > 0 else float("nan")


def evaluate_model(
    model: GPTModelV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    eval_iter: int,
) -> tuple[float, float]:
    """
    Return training and validation loss separately without any tracking or computation graph
    NOTE: optimized so that model is in inference mode runs a forward pass and the state is updated back to training.
    """
    with torch.inference_mode():
        was_training = model.training
        model.eval()
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train(was_training)
    return train_loss, val_loss


def train_classifier_simple(
    model: GPTModelV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    eval_freq: int,
    eval_iter: int,
    n_epochs: int,
):
    num_samples_seen = []
    train_losses = []
    val_losses = []
    global_step = 0
    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        for input_batch, target_batch in train_loader:
            # backprop
            model.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            # update tracking
            num_samples_seen.append(len(input_batch))
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Current epoch: {epoch}Current global step {global_step}",
                    f"train_loss: {train_loss}, validation_loss: {val_loss}",
                )
        # Calculate accuracy
        train_accuracy = calc_accuracy_loader(train_loader, model, device, eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, eval_iter)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
    return train_losses, val_losses, train_accuracies, val_accuracies, num_samples_seen


if __name__ == "__main__":
    from ch04.practice.gpt_model import GPT_CONFIG_124M

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    n_classes = 2
    torch.manual_seed(123)
    model = GPTModelV2(GPT_CONFIG_124M)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    # Train only Final Layer, Final Pre-Layer Norm, Final Transformer Block
    model.lm_head = torch.nn.Linear(GPT_CONFIG_124M.embed_dim, n_classes, False, device)
    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.layers[-1].parameters():
        param.requires_grad = True

    batch_size = 8
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset, val_dataset, test_dataset = get_train_test_dataset_split()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=4e-4, weight_decay=0.1)

    start_time = time.time()
    train_losses, val_losses, train_accuracies, val_accuracies, num_samples = (
        train_classifier_simple(
            model, train_loader, val_loader, optimizer, device, 50, 5, 5
        )
    )
    execution_time = time.time() - start_time
    print(f"Training completed in {(execution_time / 60):2f} minutes.")
