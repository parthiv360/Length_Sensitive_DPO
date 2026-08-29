import torch
import re

from experiments.baseline import Baseline
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
import os
from datetime import datetime
from pathlib import Path
import argparse
from tqdm.auto import tqdm

LOG_DIR = Path(__file__).resolve().parent.parent / "run_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"LNRS_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log"

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

class LNRSEvaluator:
    def __init__(self, model_name:str,max_length:int=512):
        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """
        Load the model and tokenizer.
        """

        logger.info("Loading tokenizer and model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=model_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if not torch.cuda.is_available():
            self.model.to(self.device)

        self.model.eval()
        logger.info("Model and tokenizer loaded successfully")

    @torch.no_grad()
    def generate_response(self,prompt,max_new_tokens:int = 128):

        inputs = self.tokenizer(
            prompt,
            return_tensors ="pt",
            truncation = True,
            max_length= self.max_length
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model.generate(
            **inputs,
            max_new_tokens= max_new_tokens,
            do_sample = False,
            temperature = None,
            pad_token_id = self.tokenizer.eos_token_id
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens = True
        ).strip()

        return answer
        


    @torch.no_grad()
    def evaluate(self, prompt:str, answer:str):
        """
        Evaluate the model on a given prompt and answer.
        """

        prompt_tokens = self.tokenizer(prompt,
                                       return_tensors="pt",
                                       add_special_tokens=False,)

        answer_tokens = self.tokenizer(answer,
                                        return_tensors="pt",
                                        add_special_tokens=False,)

        input_ids = torch.cat([prompt_tokens["input_ids"], answer_tokens["input_ids"]], dim=1).to(self.model.device)
        prompt_length = prompt_tokens["input_ids"].size(1)
        outputs = self.model(input_ids)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        answer_log_probs = log_probs[0, prompt_length-1:,:,]
        answer_labels = shift_labels[0, prompt_length-1:]

        token_log_probs = answer_log_probs.gather(1, answer_labels.unsqueeze(-1)).squeeze(-1)
        score = token_log_probs.mean().item()
        return score

    def predict(self, prompt:str, choices:list):
        """
        Predict the best choice for a given prompt.
        """

        scores = []
        for choice in choices:
            score = self.evaluate(prompt, choice)
            scores.append(score)

        prediction = max(range(len(choices)), key=lambda i: scores[i])
        return prediction, scores

    def build_ludwig_prompt(self, data):
        return (
            f"Utterance: {data['utterance']}\n"
            f"Response: {data['response']}\n"
            f"Does the response imply that the answer to the utterance is yes or no?\n"
            f"Answer:"
        )

    def ludwig_evaluate(self, dataset):
        correct = 0
        total = min(len(dataset),600)
        dataset = dataset.select(range(total))
        logger.info("Total evaluation data: %d", total)
        
        for data in tqdm(dataset, total=total, desc="Evaluating on Ludwig"):
            prompt = self.build_ludwig_prompt(data)
            choices = ["yes","no"]
            prediction, score = self.predict(prompt, choices)
            predicted_ans = choices[prediction]
            gt = data["implicature"].lower()
            if predicted_ans == gt:
                correct +=1

        accuracy = correct/total if total else 0.0
        logger.info("Total Correct: %d", correct)
        logger.info("LUDWIG accuracy: %.2f%%",accuracy * 100)

        return accuracy

    def build_pragmega_prompt(self,data):
        return (
            f"{data['content']}\n"
            f"Answer:"
        )

    def get_pragmega_gold_answer(self,data):
        prompt = data["prompt"]
        true_answer = int(data["randomized_true_answer"])

        query = re.search(
            r"(?:Options|Punchlines):\s*\n(.*?)\nAnswer:",
            prompt,
            re.DOTALL,
        )

        if not query:
            raise RuntimeError("Could not find the gold answer")

        options_text = query.group(1)
        options = re.findall(
            r"\d+\)\s*(.*?)(?=\n\d+\)|$)",
            options_text,
            re.DOTALL
        )

        if not options:
            raise RuntimeError("Could not parse options")

        if not 1 <= true_answer <= len(options):
            raise RuntimeError("Gold answer index is out of range")

        return options[true_answer-1].strip()

    def pragmega_evaluate(self,dataset):
        results = []
        total = 0

        for phenomena, data in dataset.items():
            logger.info(
                "Number of samples in phenomenon %s: %d",
                phenomena,
                len(data)
            )

            for _,row in tqdm(data.iterrows(),total = len(data), desc=f"Evaluating {phenomena}"):
                prompt = self.build_pragmega_prompt(row)
                gold_answer = self.get_pragmega_gold_answer(row)
                model_answer = self.generate_response(prompt)

                results.append({
                    "phenomena": phenomena,
                    "item_id": row["item_id"],
                    "prompt": prompt,
                    "gold_answer": gold_answer,
                    "model_answer": model_answer,
                })

                total += 1

        logger.info("Total LNRS samples: %d", total)

        for result in results[:5]:
            logger.info(
                "\nItem ID: %s"
                "\nPROMPT:\n%s"
                "\nGOLD ANSWER: %s"
                "\nMODEL ANSWER: %s"
                "\n--------------------------------",
                result["item_id"],
                result["prompt"],
                result["gold_answer"],
                result["model_answer"],
            )
        return results


    def build_social_iqa_prompt(self,data):
        return (
            f"Context: {data['context']}\n"
            f"Question: {data['question']}\n"
            f"Answer:"
        )

    def social_iqa_evaluate(self,dataset):
        results = []
        total = 0
        logger.info("Number of Social-IQA samples: %d",len(dataset))

        for data in tqdm(dataset, total=len(dataset), desc="Evaluating Social-IQA"):
            prompt = self.build_social_iqa_prompt(data)
            model_answer = self.generate_response(prompt)
            choices = {
                "1": data["answerA"],
                "2": data["answerB"],
                "3": data["answerC"],
            }
            gold_answer = choices[str(data["label"])]
            results.append({
                "prompt": prompt,
                "gold_answer": gold_answer,
                "model_answer": model_answer,
            })
            total +=1

        logger.info("Total Social_IQA samples: %d", total)
        
        for result in results[:5]:
            logger.info(
                "\nPROMPT:\n%s"
                "\nGOLD ANSWER: %s"
                "\nMODEL ANSWER: %s"
                "\n--------------------------------",
                result["prompt"],
                result["gold_answer"],
                result["model_answer"],
            )
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="allenai/open-instruct-pythia-6.9b-tulu")
    parser.add_argument("--dataset-name", type=str, default="UCL-DARK/ludwig")
    args = parser.parse_args()

    baseline = Baseline(model_name=args.model_name, dataset_name=args.dataset_name)
    baseline.load_dataset()
    dataset = baseline.dataset
    evaluator = LNRSEvaluator(model_name=args.model_name)
    evaluator.load_model()

    if args.dataset_name == "UCL-DARK/ludwig":
        logger.info("LNRS Evaluation on the ludwig dataset")
        evaluator.ludwig_evaluate(dataset["test"])
    elif args.dataset_name == "lm-pragmatics":
        logger.info("LNRS Evaluation on the Pragmega dataset")
        evaluator.pragmega_evaluate(dataset)
    elif args.dataset_name == "allenai/social_i_qa":
        logger.info("LNRS Evaluation on the Social-IQA dataset")
        evaluator.social_iqa_evaluate(dataset["validation"])
        
