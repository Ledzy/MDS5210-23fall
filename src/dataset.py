from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset, Features
from transformers import GPT2Tokenizer, GPT2TokenizerFast
import numpy as np
import datasets
import torch
from tqdm import tqdm
import random
import json
import os
from tokenizer import TiktokenTokenizer


class DahoasSFTStaticPromptsDataset(Dataset):

    def __init__(self,
                 block_size,
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Dahoas/rm-static", split="train")
        self.prompts = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading DahoasSFTStaticPromptsDataset")
        for data in dataset:
            cnt += 1
            prompt = data['prompt']
            tokens = tokenizer(prompt,
                               max_length=block_size,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")

            self.prompts.append(
                [tokens['input_ids'], tokens['attention_mask'], torch.sum(tokens['attention_mask'])])

            if max_examples and cnt >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("fka/awesome-chatgpt-prompts", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["prompt"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx][0], self.prompts[idx][1], self.prompts[idx][2]  # (1, T), (1, T)

class RLHFDataset(Dataset):
    """
    https://huggingface.co/datasets/Anthropic/hh-rlhf#dataset-summary
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        cache_dir = f"./rlhf-dataset-{split}"
        if os.path.exists(cache_dir):
            dataset = datasets.load_from_disk(cache_dir)
        else:
            dataset = load_dataset("Anthropic/hh-rlhf", split=split, data_dir="helpful-base")
            dataset.save_to_disk(cache_dir)
        self.pairs = []
        self.masks = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        torch.manual_seed(123) # ensure consistent dataset split
        num_data = len(dataset) // 2
        selected_idx_list = np.random.choice(len(dataset), num_data, replace=False)
        sub_dset = dataset[selected_idx_list]

        print(f"Loading RLHF Dataset...")
        for i in tqdm(range(num_data)):
            indices = []
            masks = []
            for split in ["chosen", "rejected"]:
                out = tokenizer(sub_dset[split][i],
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
                
                indices.append(out["input_ids"])
                masks.append(out["attention_mask"])
            self.pairs.append(torch.stack(indices, dim=0))
            self.masks.append(torch.stack(masks, dim=0))
            
            if max_examples and i >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["chosen"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)

class SFTDataset(Dataset):
    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        save = False
        if os.path.exists(f"sft_{split}.json"):
            with open(f"./sft_{split}.json") as fp:
                dataset_chosen = json.load(fp)
        else:
            save = True
            dataset = load_dataset("Anthropic/hh-rlhf", split=split, data_dir="helpful-base")

            # split half for SFT
            torch.manual_seed(123) # for consistent dataset split
            sft_idx = torch.randperm(len(dataset)//2)
            dataset = dataset[sft_idx]
            dataset_chosen = dataset["chosen"]

        self.tokens = []
        self.block_size = block_size
        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading SFT {split} split")
        for chosen in dataset_chosen:
            cnt += 1
            response_text = chosen + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens from {cnt} examples.")
        
        if save:
            import json
            json.dump(dataset_chosen, f"sft_{split}.json")

    def __len__(self):
        import sys
        return sys.maxsize

    def __getitem__(self, idx):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        return x, y

class EYLSFTStaticDataset(Dataset):

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        if split == "train":
            with open("./sft_train.json") as fp:
                dataset = json.load(fp)
        else:
            with open("./sft_test.json") as fp:
                dataset = json.load(fp)
        self.tokens = []
        self.block_size = block_size
        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading EYLSFTStaticDataset {split} split")
        for chosen in dataset:
            cnt += 1
            response_text = chosen + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens from {cnt} examples.")

    def __len__(self):
        import sys
        return sys.maxsize

    def __getitem__(self, idx):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        return x, y


class DahoasSFTStaticDataset(IterableDataset):
    """
    https://huggingface.co/datasets/Dahoas/sft-static
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset(
            "Dahoas/sft-static",
            revision="90e35d9cd625075f1224c4241734716ec9f0db78",
            split=split)
        self.tokens = []
        self.block_size = block_size

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading DahoasSFTStaticDataset {split} split")
        for data in dataset:
            cnt += 1
            prompt = data['prompt']

            response_text += prompt + data['response'] + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)

    def __iter__(self):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        yield x, y


class DahoasRMStaticDataset(Dataset):
    """
    https://huggingface.co/datasets/Dahoas/rm-static
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Dahoas/rm-static", split=split)
        self.pairs = []
        self.masks = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading DahoasRMStaticDataset {split} split")
        for data in dataset:
            cnt += 1
            prompt = data['prompt']

            positive_text = prompt + data['chosen'] + "<|endoftext|>"
            positive = tokenizer(positive_text,
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

            negative_text = prompt + data['rejected'] + "<|endoftext|>"
            negative = tokenizer(negative_text,
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

            self.pairs.append(
                torch.stack((positive['input_ids'], negative['input_ids']),
                            dim=0))

            self.masks.append(
                torch.stack(
                    (positive['attention_mask'], negative['attention_mask']),
                    dim=0))
            if max_examples and cnt >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("Dahoas/rm-static", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["prompt"] + data["chosen"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)


class AnthropicHHRLHFDataset(Dataset):
    """
    https://huggingface.co/datasets/Anthropic/hh-rlhf#dataset-summary
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        self.pairs = []
        self.masks = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        for data in dataset:
            positive = tokenizer(data["chosen"],
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
            positive_indices = positive["input_ids"]
            positive_mask = positive["attention_mask"]

            negative = tokenizer(data["rejected"],
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
            negative_indices = negative["input_ids"]
            negative_mask = negative["attention_mask"]

            self.pairs.append(
                torch.stack((positive_indices, negative_indices), dim=0))

            self.masks.append(
                torch.stack((positive_mask, negative_mask), dim=0))
            cnt += 1
            if max_examples and cnt >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["chosen"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)
