"""Image token sequences in the batch format `ptp.lit` expects.

The repo's own data path exists to pack variable-length documents together and
keep them from attending across boundaries. Image sequences are all the same
length, so none of that machinery applies -- but `make_completion_mask_mod`
indexes the document tensors unconditionally, so they cannot simply be left
out. Supplying one document that spans the whole sequence makes the document
logic a no-op and leaves plain causal attention behind.

A sequence is `[class, t_0, ... t_{S-1}]`, matching `llamagen_hf`: the class
label rides in the extended vocabulary, so completion start 1 means "predict
the first patch given only the class".
"""
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class ImageTokenDataset(Dataset):
    """Token sequences sampled from the teacher, plus the completion starts.

    Start positions are drawn per item rather than fixed, so a sequence
    contributes different completions each epoch. They avoid 0 because position
    0 holds the class and `ptp.lit` asserts every start is positive.
    """

    def __init__(self, tokens, labels, code_vocab, num_completions, completion_length,
                 prepend_label=True, left_edges=None, right_edges=None):
        self.prepend_label = prepend_label
        self.left_edges = left_edges
        self.right_edges = right_edges
        self.tokens = tokens
        self.labels = labels
        self.code_vocab = code_vocab
        self.num_completions = num_completions
        self.completion_length = completion_length

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, index):
        codes = self.tokens[index].long()
        if self.prepend_label:
            input_ids = torch.cat([
                (self.labels[index].long() + self.code_vocab).view(1),
                codes,
            ])
        else:
            # Text sequences already carry their own leading token, and every id
            # has to stay inside the model's vocabulary.
            input_ids = codes
        seq_len = input_ids.shape[0]
        high = seq_len - 1
        starts = torch.randperm(high)[:self.num_completions] + 1
        if starts.numel() < self.num_completions:  # very short sequences
            pad = torch.randint(1, seq_len, (self.num_completions - starts.numel(),))
            starts = torch.cat([starts, pad])
        item = {"input_ids": input_ids, "completion_starts": starts.sort().values}
        if self.left_edges is not None:
            # Precomputed against a frozen teacher. Supplying them sends ptp.lit
            # down a path that never calls `ar_forward` for the distribution,
            # which is what keeps a full finetune from distilling off itself.
            item["bin_edges_left"] = self.left_edges[index]
            item["bin_edges_right"] = self.right_edges[index]
        return item


def image_collate_fn(items):
    input_ids = torch.stack([item["input_ids"] for item in items])
    starts = torch.stack([item["completion_starts"] for item in items])
    batch_size, seq_len = input_ids.shape
    num_completions = starts.shape[1]
    zeros_seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    edges = {}
    if "bin_edges_left" in items[0]:
        edges = {
            "bin_edges_left": torch.stack([i["bin_edges_left"] for i in items]),
            "bin_edges_right": torch.stack([i["bin_edges_right"] for i in items]),
        }
    return {
        **edges,
        "input_ids": input_ids,
        "input_mask": None,
        "completion_starts": starts,
        "completion_length": None,  # filled by the data module
        # One document covering everything, which reduces the document logic in
        # make_completion_mask_mod to plain causal attention.
        "doc_ids": zeros_seq,
        "completion_doc_ids": torch.zeros(batch_size, num_completions, dtype=torch.long),
        "doc_starts": torch.zeros(batch_size, 1, dtype=torch.long),
        "doc_lengths": torch.full((batch_size, 1), seq_len, dtype=torch.long),
    }


class ImageTokenDataModule(LightningDataModule):
    """Serves pregenerated teacher samples from a single `.pt` file.

    The file is what `image_ptp/pregenerate.py` writes: `tokens`, `labels` and
    the cached bin edges. Only the tokens and labels are read here -- the edges
    are recomputed each step by `ptp.lit`, which keeps the whole pipeline on the
    repo's own tested path.
    """

    def __init__(self, data_path: str | Path, train_completion_len: int,
                 num_completions: int, batch_size: int, code_vocab: int = 16384,
                 val_split: int = 256, prepend_label: bool = True,
                 use_cached_bin_edges: bool = True, **kwargs):
        super().__init__()
        self.data_path = Path(data_path)
        self.train_completion_len = train_completion_len
        self.num_completions = num_completions
        self.batch_size = batch_size
        self.code_vocab = code_vocab
        self.val_split = val_split
        self.prepend_label = prepend_label
        self.use_cached_bin_edges = use_cached_bin_edges
        self.kwargs = kwargs
        self.datasets = {}

    def setup(self, stage: str | None = None) -> None:
        payload = torch.load(self.data_path, map_location="cpu")
        tokens, labels = payload["tokens"], payload["labels"]
        left = right = None
        if self.use_cached_bin_edges and "left_bin_edges" in payload:
            left, right = payload["left_bin_edges"], payload["right_bin_edges"]
        print(f"loaded {tokens.shape[0]} sequences of {tokens.shape[1]} tokens "
              f"from {self.data_path.name}"
              f"{' with frozen-teacher bin edges' if left is not None else ''}")
        split = min(self.val_split, tokens.shape[0] // 4)

        def make(lo, hi):
            return ImageTokenDataset(
                tokens[lo:hi], labels[lo:hi], self.code_vocab,
                self.num_completions, self.train_completion_len,
                prepend_label=self.prepend_label,
                left_edges=None if left is None else left[lo:hi],
                right_edges=None if right is None else right[lo:hi])

        self.datasets = {"val": make(0, split), "train": make(split, tokens.shape[0])}

    def _collate(self, items):
        batch = image_collate_fn(items)
        batch["completion_length"] = self.train_completion_len
        return batch

    def _loader(self, split, shuffle):
        return DataLoader(self.datasets[split], batch_size=self.batch_size,
                          shuffle=shuffle, collate_fn=self._collate, **self.kwargs)

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)
