import torch

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

LOG_FILE = LOG_DIR / f"mcqa_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log"

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

    def pragmega_evaluate(self,dataset):
        total_correct = 0
        total_samples = 0

        choices = ["1","2","3","4","5"]

        for p, data in dataset.items():
            logger.info("Number of samples in phenomena: %s is %d",p,len(data))

            correct = 0
            total = len(data)

            for _,row in tqdm(data.iterrows(),total=total,desc=f"Evaluating {p}"):
                prompt = row["prompt"]
                prediction, score = self.predict(prompt, choices)
                predicted_ans = prediction+1
                gt = int(row["randomized_true_answer"])

                if predicted_ans == gt:
                    correct +=1

            total_correct+=correct
            total_samples+=total
        
        accuracy = total_correct/total_samples if total_samples else 0.0
        logger.info("Total Correct: %d", total_correct)
        logger.info("PRAGMEGA accuracy: %.2f%%",accuracy * 100)

        return accuracy

    def build_social_iqa_prompt(self,data):
        return (
            f"Context: {data['context']}\n"
            f"Question: {data['question']}\n"
            f"Answer choices:\n"
            f"1. {data['answerA']}\n"
            f"2. {data['answerB']}\n"
            f"3. {data['answerC']}\n"
            f"Answer:"
        )

    def social_iqa_evaluate(self,dataset):
        correct = 0
        total = len(dataset)
        choices = ["1","2","3"]
        logger.info("Total Evaluation Data %d", total)

        for data in tqdm(dataset, total=total, desc="Evaluating Social-IQA"):
            prompt = self.build_social_iqa_prompt(data)
            prediction, score = self.predict(prompt,choices)
            predicted_ans = int(choices[prediction])
            gt = int(data["label"])

            if predicted_ans == gt :
                correct +=1

        accuracy = correct/total if total else 0.0
        logger.info("Total Correct: %d", correct)
        logger.info("SOCIAL-IQA accuracy: %.2f%%",accuracy * 100)

        return accuracy
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
        logger.info("MCQA Evaluation on the ludwig dataset")
        evaluator.ludwig_evaluate(dataset["test"])
    elif args.dataset_name == "lm-pragmatics":
        logger.info("MCQA Evaluation on the Pragmega dataset")
        for phenomenon, data in dataset.items():
            print("\n==============================")
            print(phenomenon)
            print(data.columns.tolist())
            print(data.iloc[0].to_dict())
    elif args.dataset_name == "allenai/social_i_qa":
        logger.info("MCQA Evaluation on the Social-IQA dataset")
        evaluator.social_iqa_evaluate(dataset["validation"])
        
