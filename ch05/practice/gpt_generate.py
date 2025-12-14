from itertools import islice
from pathlib import Path

import tiktoken
import torch
from tiktoken import Encoding
from torch.utils.data.dataloader import DataLoader

from ch04.practice.gpt_model import (
    GPT_CONFIG_124M,
    GPTModelV2,
    generate_simple_text_with_cache,
)
from ch05.practice.data_tools import create_dataloader_v1


def text_to_tokens(text: str, tokenizer: Encoding, device: str) -> torch.Tensor:
    return torch.tensor(
        tokenizer.encode(text), dtype=torch.long, device=device
    ).unsqueeze(0)


def tokens_ids_to_text(input_ids: torch.Tensor, tokenizer: Encoding) -> str:
    return tokenizer.decode(input_ids.squeeze(0).tolist())


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModelV2,
    device: str,
) -> torch.Tensor:
    input_batch = input_batch.to(device, dtype=torch.long)
    target_batch = target_batch.to(device, dtype=torch.long)
    # (B, S, V)
    logits = model(input_batch)
    _, _, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)  # (B*S, V)
    target_flat = target_batch.reshape(-1)  # (B*S,)
    breakpoint()
    loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)
    return loss


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


def generate_and_print_sample(
    model: GPTModelV2, tokenizer: tiktoken.Encoding, device: str, prompt: str
) -> None:
    max_seq_len = model.pos_emb.weight.shape[0]
    input_ids = text_to_tokens(prompt, tokenizer).to(device)

    with torch.inference_mode():
        was_training = model.training
        model.eval()
        output_ids = generate_simple_text_with_cache(model, input_ids, 50, max_seq_len)
        model.train(was_training)

    output_text = tokens_ids_to_text(output_ids, tokenizer)
    print(output_text.replace("\n", " "))


def train_model_simple(
    model: GPTModelV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: tiktoken.Encoding,
    n_epochs: int,
) -> tuple[list, list, list]:
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    # Move this out?
    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            train_loss = calc_loss_batch(input_batch, target_batch, model, device)
            train_loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Global step {global_step:06d}): "
                    f"Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}"
                )
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


# Non-greedy sampling
@torch.no_grad()
def generate(
    model: GPTModelV2,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_len: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    model.reset_cache()
    output_ids = input_ids
    for i in range(max_new_tokens):
        if use_cache and i != 0:
            context = output_ids[:, -1:]
        else:
            context = output_ids[:, -context_len:]
        logits = model(context, use_cache=use_cache)
        next_token_logits = logits[:, -1, :]
        if top_k:
            top_k_vals, _ = torch.topk(next_token_logits, top_k)
            next_token_logits = torch.where(
                next_token_logits < top_k_vals[:, -1],
                torch.tensor(-torch.inf).to(logits.device),
                next_token_logits,
            )
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = next_token_logits.argmax(dim=-1, keepdim=True)

        if next_id == eos_id:
            break

        output_ids = torch.cat([output_ids, next_id], dim=1)
    return output_ids


def test_training_loop(
    model: GPTModelV2,
    tokenizer: tiktoken.Encoding,
    train_loader: DataLoader,
    val_loader: DataLoader,
    start_context: str,
    device: str,
) -> tuple[list, list, list]:
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=4e-4, weight_decay=0.1)
    train_loss, val_loss, track_tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        eval_freq=5,
        eval_iter=5,
        start_context=start_context,
        tokenizer=tokenizer,
        n_epochs=10,
    )
    return train_loss, val_loss, track_tokens_seen


def test_nongreedy_generate(
    model: GPTModelV2, tokenizer: tiktoken.Encoding, device: str
) -> None:
    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        input_ids=text_to_tokens("Every effort moves you", tokenizer, device=device),
        max_new_tokens=15,
        context_len=GPT_CONFIG_124M.max_seq_len,
        temperature=1.4,
        top_k=25,
    )
    print(f"Output text:\n {tokens_ids_to_text(token_ids, tokenizer)}")


if __name__ == "__main__":
    torch.manual_seed(123)
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    model = GPTModelV2(cfg=GPT_CONFIG_124M).to(device)

    proj_dir = Path("/workspace")
    filepath = proj_dir / "ch02/01_main-chapter-code" / "the-verdict.txt"
    with open(filepath, "r") as f:
        text_data = f.read()
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        text=train_data,
        batch_size=2,
        max_seq_len=GPT_CONFIG_124M.max_seq_len,
        stride=GPT_CONFIG_124M.max_seq_len,
        drop_last=True,
        shuffle=False,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        text=val_data,
        batch_size=2,
        max_seq_len=GPT_CONFIG_124M.max_seq_len,
        stride=GPT_CONFIG_124M.max_seq_len,
        drop_last=True,
        shuffle=False,
        num_workers=2,
    )

    train_loss, val_loss, track_tokens_seen = test_training_loop(
        model, tokenizer, train_loader, val_loader, start_context, device
    )
    test_nongreedy_generate(model, tokenizer, device)
