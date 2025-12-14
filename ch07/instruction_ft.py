import json
import os
import urllib.request

import tiktoken
import torch
from torch.utils.data import Dataset

INSTRUCTION_DATASET_PATH = "instruction-data.json"
URL_PATH = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)


def download_and_load_file(file_path: str, url: str) -> list[dict[str, str]]:
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# Change to Qwen Instruction formatting
def format_input(entry: dict[str, str]) -> tuple[str, str]:
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    response_text = f"\n\\n### Response:\\n{entry['output']}"

    return instruction_text + input_text, response_text


class InstructionDatasetV1(Dataset):
    def __init__(
        self,
        data: list[dict[str, str]],
        tokenizer: tiktoken.Encoding,
        max_seq_len: int = 1024,
        pad_id: int = 50256,
    ) -> None:
        self.input_ids, self.target_ids = [], []
        for entry in data:
            prompt, response = format_input(entry)
            prompt_token_ids, response_token_ids = (
                tokenizer.encode(prompt),
                tokenizer.encode(response),
            )
            prompt_pad_len = max_seq_len - len(prompt_token_ids)
            if prompt_pad_len < 0:
                prompt_token_ids.extend([pad_id] * prompt_pad_len)
            else:
                prompt_token_ids = prompt_token_ids[:max_seq_len]
            response_pad_len = max_seq_len - len(response_token_ids)
            if response_pad_len < 0:
                response_token_ids.extend([pad_id] + [-100] * (response_pad_len - 1))
            else:
                response_token_ids = response_token_ids[:max_seq_len]
            self.input_ids.append(torch.tensor(prompt_token_ids))
            self.target_ids.append(torch.tensor(response_token_ids))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]

    def __len__(self) -> int:
        return len(self.input_ids)


class InstructionDataset(Dataset):
    def __init__(
        self, data: list[dict[str, str]], tokenizer: tiktoken.Encoding
    ) -> None:
        self.data = data
        self.token_ids = []
        for entry in data:
            prompt, response = format_input(entry)
            self.token_ids.append(tokenizer.encode(prompt + response))

    def __getitem__(self, index: int) -> list[int]:
        return self.token_ids[index]

    def __len__(self) -> int:
        return len(self.token_ids)


def custom_collate_draft_1(
    batch: list[list[int]], pad_token_id: int = 50256, device: str = "cpu"
) -> torch.Tensor:
    batch_max_len = max(len(item) + 1 for item in batch)
    batch_ids = []

    for sample in batch:
        new_sample = sample.copy()
        new_sample += [pad_token_id]
        new_sample.extend([pad_token_id] * (batch_max_len - len(sample)))
        new_sample = new_sample[:-1]
        batch_ids.append(torch.tensor(new_sample[:-1]))

    return torch.stack(batch_ids).to(device)


def custom_collate_fn(
    batch: list[list[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    device: str = "cpu",
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_max_len = min(
        max_seq_len + 1 if max_seq_len else -1, max(len(item) + 1 for item in batch)
    )
    input_ids, target_ids = [], []

    for seq in batch:
        seq_len = min(len(seq), batch_max_len - 1)
        pad_len = batch_max_len - seq_len
        input = seq[:seq_len] + [pad_token_id] * pad_len
        target = seq[1:seq_len] + [pad_token_id] + [ignore_index] * pad_len
        input_ids.append(input)
        target_ids.append(target)

    return torch.tensor(input_ids, device=device), torch.tensor(
        target_ids, device=device
    )


def setup_train_test_split(
    split_ratio: float = 0.85, test_ratio: float = 0.05
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    data = download_and_load_file(INSTRUCTION_DATASET_PATH, URL_PATH)
    total_char = len(data)
    #


if __name__ == "__main__":
    data = download_and_load_file(INSTRUCTION_DATASET_PATH, URL_PATH)
    print("Number of entries:", len(data))
    from functools import partial

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        max_seq_len=1024,
    )
