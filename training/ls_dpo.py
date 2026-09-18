# This file contains the code for Length-Sensitive DPO. 

from training.dpo import DPOTrainer, DPO, DPODataCollator
import torch
import re
import numpy as np
from experiments.baseline import Baseline
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset, load_dataset, concatenate_datasets
import logging
import os
from datetime import datetime
from pathlib import Path
import argparse
from tqdm.auto import tqdm
import pandas as pd
import wandb

LOG_DIR = Path(__file__).resolve().parent.parent / "run_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"DPO_Training_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
    force=True
)

logger = logging.getLogger(__name__)

class RDPOTrainer(DPOTrainer):
    def __init__(self, ref_model=None, beta=0.1, alpha = 0.01, *args, **kwargs):
        super().__init__(ref_model, beta, *args, **kwargs)
        self.alpha = alpha

        logger.info(f"Initialized LS-DPO Trainer: " f"beta={self.beta}, alpha={self.alpha}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        chosen_labels = inputs["chosen_labels"]

        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        rejected_labels = inputs["rejected_labels"]

        chosen_logs = self.get_log_probs(model,chosen_input_ids,chosen_attention_mask,chosen_labels)
        rejected_logs = self.get_log_probs(model,rejected_input_ids,rejected_attention_mask,rejected_labels)

        with torch.no_grad():
            ref_chosen_logs = self.get_log_probs(self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels)
            ref_rejected_logs = self.get_log_probs(self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels)

        log_ratios = chosen_logs-rejected_logs
        ref_log_ratios = (ref_chosen_logs-ref_rejected_logs)
        dpo_logits = self.beta*(log_ratios - ref_log_ratios)

        chosen_lengths = (chosen_labels !=-100).sum(dim=1)
        rejected_lengths = (rejected_labels!=-100).sum(dim=1)
        length_difference = (chosen_lengths-rejected_lengths)
        length_reg = (self.alpha * length_difference)

        logits = (dpo_logits+length_reg)
        loss = -torch.nn.functional.logsigmoid(logits).mean()

        if return_outputs:
            return loss , {
                "chosen_logs": chosen_logs,
                "rejected_logs": rejected_logs,
                "ref_chosen_logs": ref_chosen_logs,
                "ref_rejected_logs": ref_rejected_logs,
                "dpo_logits": dpo_logits,
                "length_regularization": length_reg,
                "chosen_lengths": chosen_lengths,
                "rejected_lengths": rejected_lengths,
            }
        return loss

class RDPO(DPO):
    def train(self,dataset,args):
        training_args = TrainingArguments(
        output_dir=self.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim = args.optimizer,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns = False,
        report_to="wandb" if args.use_wandb else "none")
        
        data_collator = DPODataCollator(tokenizer=self.tokenizer)
        
        trainer = RDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            beta=self.beta,
            alpha=self.alpha,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        if args.use_wandb:
            wandb.init(
                project="length-sensitive-dpo",
                name=f"LS_DPO-{self.model_name.split('/')[-1]}",
                config={
                    "model": self.model_name,
                    "datasets": args.dataset_name,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "effective_batch_size": (
                        args.batch_size * args.gradient_accumulation_steps
                    ),
                    "epochs": args.num_train_epochs,
                    "max_length": args.max_length,
                    "optimizer": args.optimizer,
                    "warmup_steps": args.warmup_steps,
                },
            )
        logger.info("Starting training...")
        trainer.train()
        if args.use_wandb:
            wandb.finish()
        logger.info("Training completed. Saving model to %s", self.output_dir)
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LS_DPO Trainer")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--optimizer", type=str, default="adamw_torch", help="Optimizer to use for training")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_steps", type=int, default=-1, help="Total number of training steps to perform. If set to -1, it will be determined by the number of epochs.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "epoch", "steps"], help="The checkpoint save strategy to use.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to save. Older checkpoints will be deleted.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory at the cost of slower backward pass.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--use_wandb", action='store_true', help="Whether to use Weights & Biases for logging.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")

    args = parser.parse_args()

    ls_dpo = RDPO(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
    )

    ls_dpo.load_model()
    datasets = args.dataset_name.split(",")
    train_dataset = ls_dpo.concatenate_dataset(datasets)
    train_dataset = train_dataset.filter(lambda x: x["prompt"] is not None
                                            and x["chosen"] is not None
                                            and x["rejected"] is not None
                                            and isinstance(x["prompt"], str)
                                            and isinstance(x["chosen"], str)
                                            and isinstance(x["rejected"], str)
                                            and x["prompt"].strip() != ""
                                            and x["chosen"].strip() != ""
                                            and x["rejected"].strip() != "")
    train_dataset = train_dataset.map(ls_dpo.tokenize_data, remove_columns=train_dataset.column_names)

    ls_dpo.train(train_dataset, args)
    