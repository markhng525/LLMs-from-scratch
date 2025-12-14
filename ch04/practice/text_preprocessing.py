from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __getitem__(self, index) -> Dataset:
        return super().__getitem__(index)
