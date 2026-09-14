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

class DPODataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        chosen = {
            "input_ids": [f["chosen_input_ids"] for f in features],
            "attention_mask": [f["chosen_attention_mask"] for f in features],
            "labels": [f["chosen_labels"] for f in features],
        }

        rejected = {
            "input_ids": [f["rejected_input_ids"] for f in features],
            "attention_mask": [f["rejected_attention_mask"] for f in features],
            "labels": [f["rejected_labels"] for f in features],
        }

        chosen_batch = self._pad(chosen)
        rejected_batch = self._pad(rejected)

        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "chosen_labels": chosen_batch["labels"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "rejected_labels": rejected_batch["labels"],
        }

    def _pad(self, features):
        max_length = max(len(x) for x in features["input_ids"])

        input_ids = []
        attention_mask = []
        labels = []

        for ids, mask, lbls in zip(features["input_ids"],features["attention_mask"],features["labels"],):
            padding_length = max_length - len(ids)

            input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)

            attention_mask.append(mask + [0] * padding_length)

            labels.append(lbls + [-100] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
class DPOTrainer(Trainer):
    def __init__(self, ref_model= None, beta= 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta

        self.ref_model.eval()

        for param in self.ref_model.parameters():
            param.requires_grad = False


    def get_log_probs(self, model, input_ids, attention_mask, labels):
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[:,:-1,:]
        shift_labels = labels[:,1:]
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        gather_labels = shift_labels.clone()
        gather_labels[gather_labels == -100] = 0
        token_log_probs = torch.gather(log_probs, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
        mask = shift_labels != -100
        token_log_probs = token_log_probs * mask
        seq_log_probs = token_log_probs.sum(dim=-1)
        return seq_log_probs

    def compute_dpo_loss(self, chosen_log_probs, rejected_log_probs, ref_chosen_log_probs, ref_rejected_log_probs):
        log_ratios = (chosen_log_probs - rejected_log_probs)
        ref_log_ratios = (ref_chosen_log_probs - ref_rejected_log_probs)

        logits = self.beta * (log_ratios - ref_log_ratios)
        loss = -torch.nn.functional.logsigmoid(logits).mean()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        chosen_labels = inputs["chosen_labels"]

        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        rejected_labels = inputs["rejected_labels"]

        chosen_logs = self.get_log_probs(model, chosen_input_ids, chosen_attention_mask, chosen_labels)
        rejected_logs = self.get_log_probs(model, rejected_input_ids, rejected_attention_mask, rejected_labels)

        with torch.no_grad():
            ref_chosen_logs = self.get_log_probs(self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels)
            ref_rejected_logs = self.get_log_probs(self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels)

        loss = self.compute_dpo_loss(chosen_logs, rejected_logs, ref_chosen_logs, ref_rejected_logs)

        if return_outputs:
            return loss, {
                "chosen_logs": chosen_logs,
                "rejected_logs": rejected_logs,
                "ref_chosen_logs": ref_chosen_logs,
                "ref_rejected_logs": ref_rejected_logs
            }
        return loss
    

class DPO:
    def __init__(self, model_name, output_dir, max_length=512, beta =0.1):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.beta = beta

        self.tokenizer = None
        self.model = None
        self.ref_model = None   
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.ref_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.ref_model.eval()

        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.model = self.model.to(self.device)
        self.ref_model = self.ref_model.to(self.device)

        logger.info("Both Models and tokenizer loaded successfully")
        
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
            max_length= self.max_length)

        chosen_tokens = self.tokenizer(
            " " + data["chosen"],
            add_special_tokens=False,
        )

        rejected_tokens = self.tokenizer(
            " " + data["rejected"],
            add_special_tokens=False,
        )

        prompt_ids = prompt_tokens["input_ids"]
        max_chosen_length = self.max_length - len(prompt_ids) -1
        max_rejected_length = self.max_length - len(prompt_ids) -1

        chosen_ids = chosen_tokens["input_ids"][:max_chosen_length]
        rejected_ids = rejected_tokens["input_ids"][:max_rejected_length]

        chosen_input_ids = prompt_ids + chosen_ids + [self.tokenizer.eos_token_id]
        rejected_input_ids = prompt_ids + rejected_ids + [self.tokenizer.eos_token_id]

        chosen_labels = [-100] * len(prompt_ids) + chosen_ids + [self.tokenizer.eos_token_id]
        rejected_labels = [-100] * len(prompt_ids) + rejected_ids + [self.tokenizer.eos_token_id]

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": [1] * len(chosen_input_ids),
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": [1] * len(rejected_input_ids),
            "rejected_labels": rejected_labels,
        }
    
    def format_social_iqa(self, dataset):
        prompts = []
        chosen = []
        rejected = []

        for data in dataset:
            choices = {
                "1": data["answerA"],
                "2": data["answerB"],
                "3": data["answerC"],
            }

            gold_label = str(data["label"])
            gold_answer = choices[gold_label]

            prompt = (
                f"Context: {data['context']}\n"
                f"Question: {data['question']}\n"
                f"Answer:"
            )

            for label, other_ans in choices.items():
                if label == gold_label:
                    continue
                prompts.append(prompt)
                chosen.append(gold_answer)
                rejected.append(other_ans)

        return Dataset.from_dict({
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
        })

    def format_pub(self, dataset):
        prompts = []
        chosen = []
        rejected = []

        for data in dataset:
            options = data["options"]
            gold_answer = data["correct answer"]

            prompt = data["pretext"]

            for other_ans in options:
                if other_ans == gold_answer:
                    continue
                prompts.append(prompt)
                chosen.append(gold_answer)
                rejected.append(other_ans)

        return Dataset.from_dict({
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
        })

    def concatenate_dataset(self, datasets):
        logger.info("Concatenating datasets...")
        formatted_datasets = []
        for dataset_name in datasets:
            self.load_dataset(dataset_name)
            dataset = self.dataset

            if dataset_name == "allenai/social_i_qa":
                formatted_dataset = self.format_social_iqa(dataset["train"])
            elif dataset_name == "cfilt/PUB":
                formatted_dataset = self.format_pub(dataset["train"])
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
            remove_unused_columns = False,
            report_to="wandb" if args.use_wandb else "none")

        data_collator = DPODataCollator(tokenizer=self.tokenizer)

        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            beta=self.beta,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        if args.use_wandb:
            wandb.init(
                project="length-sensitive-dpo",
                name=f"DPO-{self.model_name.split('/')[-1]}",
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
    parser = argparse.ArgumentParser(description="DPO Trainer")
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

    dpo = DPO(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length
    )

    dpo.load_model()
    datasets = args.dataset_name.split(",")
    train_dataset = dpo.concatenate_dataset(datasets)
    train_dataset = train_dataset.filter(lambda x: x["prompt"] is not None
                                         and x["chosen"] is not None
                                         and x["rejected"] is not None
                                         and isinstance(x["prompt"], str)
                                         and isinstance(x["chosen"], str)
                                         and isinstance(x["rejected"], str)
                                         and x["prompt"].strip() != ""
                                         and x["chosen"].strip() != ""
                                         and x["rejected"].strip() != "")
    train_dataset = train_dataset.map(dpo.tokenize_data, remove_columns=train_dataset.column_names)

    dpo.train(train_dataset, args)
    


