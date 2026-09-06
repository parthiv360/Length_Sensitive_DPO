import torch
import re
import numpy as np
from experiments.baseline import Baseline
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets
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

LOG_FILE = LOG_DIR / f"SFT_Training_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log"

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

class SFTTrainer:
    def __init__(self, model_name, output_dir, max_length=512):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length

        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def load_dataset(self, dataset_name=None):

        """
        Load the dataset using the Hugging Face datasets library.
        """

        if dataset_name is not None:
            self.dataset_name = dataset_name

        logger.info("Loading dataset: %s", self.dataset_name)
        if self.dataset_name == "allenai/social_i_qa":
            logger.info("Loading SocialIQA dataset from Parquet conversion")
            
            self.dataset = load_dataset(
            "allenai/social_i_qa",
            revision="refs/convert/parquet",
        )
        elif self.dataset_name == "UCL-DARK/ludwig":
            logger.info("Loading LUDWIG dataset from Parquet conversion")

            self.dataset = load_dataset(
                "UCL-DARK/ludwig",
                revision="refs/convert/parquet",
            )
        elif self.dataset_name == "lm-pragmatics":
            logger.info("Loading Pragmega dataset")
            pragmega_dir = Path("/scratch/compuling/pasa00007/HF_DATA/datasets/lm-pragmatics/prompts")
            phenomena = [
                "Deceits",
                "IndirectSpeech",
                "Irony",
                "Maxims",
                "Metaphor",
                "Humour",
            ]
            self.dataset = {}
            for p in phenomena:
                file_path = (pragmega_dir/ f"{p}_prompts_seed0_examples0.csv")
                self.dataset[p] = pd.read_csv(file_path)

        else:
            self.dataset = load_dataset(self.dataset_name, cache_dir="/scratch/compuling/pasa00007/HF_DATA/datasets")
        logger.info("Dataset loaded successfully")

    def tokenize_data(self, data):
        prompt_tokens = self.tokenizer(
            data["prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length= 256)

        answer_tokens = self.tokenizer(
            " " + data["answer"],
            add_special_tokens=False,
            )

        input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"] + [self.tokenizer.eos_token_id]
        labels = ([-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"] + [self.tokenizer.eos_token_id])

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }
    
    def format_social_iqa(self, data):
        choices = {
            "1": data["answerA"],
            "2": data["answerB"],
            "3": data["answerC"]
        }

        gold_answer = choices[str(data["label"])]

        return {
            "prompt": (
                f"Context: {data['context']}\n"
                f"Question: {data['question']}\n"
                f"Answer:"
            ),
            "answer": gold_answer
        }

    def format_pub(self, data):
        return {
            "prompt": (
                f"{data['pretext']}\n"
                f"Answer:"
            ),
            "answer": data["correct answer"],
        }

    def concatenate_dataset(self, datasets):
        logger.info("Concatenating datasets...")
        formatted_datasets = []
        for dataset_name in datasets:
            self.load_dataset(dataset_name)
            dataset = self.dataset

            if dataset_name == "allenai/social_i_qa":
                formatted_dataset = dataset["train"].map(
                    self.format_social_iqa,
                    remove_columns=dataset["train"].column_names,
                )
            elif dataset_name == "cfilt___pub":
                formatted_dataset = dataset["train"].map(
                    self.format_pub,
                    remove_columns=dataset["train"].column_names,
                )
            else:
                raise ValueError(f"Unsupported dataset for concatenation: {dataset_name}")

            formatted_datasets.append(formatted_dataset)

        if not formatted_datasets:
            raise ValueError("At least one dataset is required for concatenation")

        train_dataset = concatenate_datasets(formatted_datasets)
        logger.info("Datasets concatenated successfully. Total samples: %d", len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=42)
        return train_dataset
        
    def train(self, dataset, args):
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
            report_to="wandb" if args.use_wandb else "none")

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            padding=True, 
            label_pad_token_id=-100,
            return_tensors="pt")

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        if args.use_wandb:
            wandb.init(
                project="length-sensitive-dpo",
                name=f"SFT-{self.model_name.split('/')[-1]}",
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
    parser = argparse.ArgumentParser(description="SFT Trainer")
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

    trainer = SFTTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length
    )

    trainer.load_model()
    datasets = args.dataset_name.split(",")
    train_dataset = trainer.concatenate_dataset(datasets)
    train_dataset = train_dataset.map(trainer.tokenize_data, remove_columns=train_dataset.column_names)

    trainer.train(train_dataset, args)
    


