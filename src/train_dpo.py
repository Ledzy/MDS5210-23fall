import click
import torch
import copy
import os
from trainers import DPOTrainer
from configs import get_configs
from gpt import GPTActor, GPTRewardModel, GPTCritic, GPT
from dataset import DahoasSFTStaticPromptsDataset, RLHFDataset


def train(pretrain, batch_size, exp_name):
    cfg = get_configs("gpt2-medium/dropout")
    cfg.max_steps = 200000 // batch_size
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    assert pretrain == "huggingface"
    cfg.exp_name = exp_name

    # model_cache_path = f"./{cfg.model_name}"
    # if os.path.exists(model_cache_path):
    #     model = GPT.from_pretrained(model_cache_path)
    model = GPT.from_pretrained(cfg)
    ref_model = GPT.from_pretrained(cfg)

    train_ds = RLHFDataset(block_size=256, split="train", max_examples=None, tokenizer_name="tiktoken/gpt2")
    test_ds = RLHFDataset(block_size=256, split="test", max_examples=None, tokenizer_name="tiktoken/gpt2")
    trainer = DPOTrainer(cfg, model, ref_model, train_ds, test_ds, beta=1.)
    trainer.fit()


@click.command()
@click.option('--strategy', '-s')
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
def main(strategy, pretrain, batch_size, exp_name):
    torch.manual_seed(1234)
    train(pretrain, batch_size, exp_name)



if __name__ == "__main__":
    main()
