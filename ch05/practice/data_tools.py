import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from ch04.practice.gpt_model import GPT_CONFIG_124M


class GPT2Dataset(Dataset):
    """This dataset returns the token ids for inputs and targets"""

    def __init__(
        self, text: str, tokenizer: tiktoken.Encoding, max_seq_len: int, stride: int
    ):
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        total_len = len(token_ids)
        for i in range(0, total_len - max_seq_len, stride):
            self.input_ids.append(torch.Tensor(token_ids[i : i + max_seq_len]))
            self.target_ids.append(torch.Tensor(token_ids[i + 1 : i + 1 + max_seq_len]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
    text: str,
    batch_size: int,
    max_seq_len: int,
    stride: int,
    drop_last: bool,
    shuffle: bool,
    num_workers: int,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPT2Dataset(text, tokenizer, max_seq_len, stride)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    from pathlib import Path

    proj_dir = Path("/workspace")
    filepath = proj_dir / "ch02/01_main-chapter-code" / "the-verdict.txt"
    with open(filepath, "r") as f:
        text_data = f.read()
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    tokenizer = tiktoken.get_encoding("gpt2")

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
        batch_size=4,
        max_seq_len=GPT_CONFIG_124M.max_seq_len,
        stride=GPT_CONFIG_124M.max_seq_len,
        drop_last=True,
        shuffle=False,
        num_workers=2,
    )
    for x, y in train_loader:
        print(x.shape, y.shape)
