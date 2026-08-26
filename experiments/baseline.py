from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

LOG_DIR = Path(__file__).resolve().parent.parent / "run_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"baseline_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


class Baseline:
    def __init__(self, model_name: str, dataset_name: str, max_length: int = 512):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_length = max_length

        self.tokenizer = None
        self.model = None
        self.dataset = None

    def load_dataset(self):
        """
        Load the dataset using the Hugging Face datasets library.
        """

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
        

    def load_model(self):
        """
        Load the model and tokenizer.
        """

        logger.info("Loading tokenizer and model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        logger.info("Model and tokenizer loaded successfully")
        

    def setup(self):
        """
        Load the dataset and model.
        """

        logger.info("Starting baseline setup")
        self.load_dataset()
        self.load_model()
        logger.info("Baseline setup completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--dataset_name", type=str, default="allenai/social_i_qa")
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()

    baseline = Baseline(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        max_length=args.max_length,
    )

    baseline.setup()
    logger.info(
        "Loaded model: %s and dataset: %s",
        baseline.model_name,
        baseline.dataset_name,
    )