import torch
import re
import numpy as np
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
    def __init__(self, model_name:str,max_length:int=512, judge_model_name: str = "Qwen/Qwen2.5-32B-Instruct",):
        self.model_name = model_name
        self.max_length = max_length
        self.judge_model_name = judge_model_name

        self.tokenizer = None
        self.model = None

        self.judge_tokenizer = None
        self.judge_model = None
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

    def load_judge_model(self):
        """
        Load the judge model and tokenizer.
        """
        logger.info("Loading judge model and tokenizer: %s", self.judge_model_name)
        self.judge_tokenizer = AutoTokenizer.from_pretrained(self.judge_model_name)

        self.judge_model = AutoModelForCausalLM.from_pretrained(
            self.judge_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.judge_model.eval()
        logger.info("Judge model and tokenizer loaded successfully")

    @torch.no_grad()
    def generate_judge_response(self, prompt, max_new_tokens: int = 128):
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = self.judge_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.judge_tokenizer(
            text,
            return_tensors="pt",
        )

        judge_device = self.judge_model.get_input_embeddings().weight.device

        inputs = {
            key: value.to(judge_device)
            for key, value in inputs.items()
        }

        outputs = self.judge_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.judge_tokenizer.eos_token_id,
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        response = self.judge_tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return response

    @torch.no_grad()
    def generate_response(self,prompt,max_new_tokens:int = 128):

        inputs = self.tokenizer(
            prompt,
            return_tensors ="pt",
            truncation = True,
            max_length= self.max_length
        )

        model_device = self.model.get_input_embeddings().weight.device

        inputs = {
            key: value.to(model_device)
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
        
    def build_judge_prompt(self, question, model_ans, gold_ans, rev=False):
        if not rev:
            first_label = "Model's Answer"
            first_answer = model_ans

            second_label = "Gold Answer for Reference"
            second_answer = gold_ans
        else:
            first_label = "Gold Answer for Reference"
            first_answer = gold_ans

            second_label = "Model's Answer"
            second_answer = model_ans
        return f"""
        [Scenario]:
        {question} 

        [{first_label}]:
        {first_answer}

        [{second_label}]:
        {second_answer}

        [System]:
                We request your evaluation of the AI
        model's answer in relation to the provided
        scenario and the gold answer. Assess the
        responses based on the following criteria:
        1. Social Understanding: How well does the
        model's answer grasp the social dynamics
        or pragmatic nuances of the scenario?
        2. Appropriateness: Is the model's answer
        appropriate and contextually fitting for the
        scenario?
        3. Insightfulness: Does the answer
        demonstrate a deep understanding of the
        underlying intentions, implicature, deceit,
        irony, sarcasm, humor, metaphor, etc.?
        4. Completeness: How comprehensive
        is the model's response in capturing the
        essential elements of the scenario?
        Please first output a single line containing
        only two numeric values representing
        scores for the model's answer and the gold
        answer respectively, on a scale of 1 to
        10, where a higher score indicates better
        performance. The two score values should
        be separated by a space. The gold answer is
        for reference only and should not strictly
        limit the evaluation.
        In the next line, provide a comprehensive
        explanation of your evaluation, discussing
        each of the criteria mentioned. This
        explanation should avoid any potential bias
        and ensure that the judgment is solely based
        on the response's merits in the context
        of the scenario and the gold answer for
        reference.
        """

    def judge_response(self, question, gold_resp, model_resp, rev=False):
        
        prompt = self.build_judge_prompt(question,model_resp,gold_resp,rev)
        text = self.generate_judge_response(prompt)
        logger.info("Judge Response:\n%s", text)
        
        lines = text.splitlines()
        if not lines:
            raise ValueError("Judge response is empty")

        first_line = lines[0].strip()
        numbers = re.findall(r"\d+(?:\.\d+)?",first_line)

        if len(numbers) < 2:
            raise ValueError("Could not find two numeric scores in the judge response")

        first_score = float(numbers[0])
        second_score = float(numbers[1])

        if not( 1 <= first_score <= 10) or not (1 <= second_score <= 10):
            raise ValueError("Scores are out of the expected range of 1 to 10")

        if not rev:
            model_score = first_score
            gold_score = second_score
        else:
            model_score = second_score
            gold_score = first_score

        return model_score, gold_score


    def get_judge_score(self, question, gold_resp, model_resp):
        model_score_1, gold_score_1 = self.judge_response(question,gold_resp,model_resp,False)
        model_score_2, gold_score_2 = self.judge_response(question, gold_resp, model_resp,True)
        model_score = (model_score_1+model_score_2)/2
        gold_score = (gold_score_1+gold_score_2)/2

        return model_score, gold_score
    
    def calculate_lnrs(self, results, tau = 1.0):
        T = len(results)
        model_score_sum = sum(result["model_score"] for result in results)
        gold_score_sum = sum(result["gold_score"] for result in results)
        model_length_sum = sum(result["model_length"] for result in results)
        gold_length_sum = sum(result["gold_length"] for result in results)

        relative_score = (model_score_sum/gold_score_sum if gold_score_sum > 0 else 0)
        length_diff = (gold_length_sum - model_length_sum)
        lf = 1/ ( 1+ np.exp(-length_diff / (tau * T)))
        lnrs = relative_score*lf    

        return lnrs

    def get_tokens_length(self, text):
        tokens = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        return len(tokens)["input_ids"]
    
    def evaluate(self, question, gold_resp, model_resp):
        model_score, gold_score = self.get_judge_score(question,gold_resp,model_resp)
        model_length = self.get_tokens_length(model_resp)
        gold_length = self.get_tokens_length(gold_resp)

        return {
            "model_answer": model_resp,
            "gold_answer": gold_resp,
            "model_score": model_score,
            "gold_score": gold_score,
            "model_length": model_length,
            "gold_length": gold_length
        }
    def build_ludwig_prompt(self, data):
        return (
            f"Utterance: {data['utterance']}\n"
            f"Response: {data['response']}\n"
            f"Does the response imply that the answer to the utterance is yes or no?\n"
            f"Answer:"
        )

    def ludwig_evaluate(self, dataset):
        results = []
        total = min(len(dataset),600)
        dataset = dataset.select(range(total))
        logger.info("Total evaluation data: %d", total)
        
        for data in tqdm(dataset, total=total, desc="Evaluating on Ludwig"):
            prompt = self.build_ludwig_prompt(data)
            model_answer = self.generate_response(prompt)
            gold_answer = data["implicature"]
            evaluation = self.evaluate(prompt, gold_answer, model_answer)
            results.append({
                "id": data["id"],
                "prompt": prompt,
                **evaluation
            })

        for result in results[:5]:
            logger.info(
                "\nID: %s"
                "\nPROMPT:\n%s"
                "\nGOLD ANSWER: %s"
                "\nMODEL ANSWER: %s"
                "\nGOLD SCORE: %f"
                "\nMODEL SCORE: %f"
                
                "\n--------------------------------",
                result["id"],
                result["prompt"],
                result["gold_answer"],
                result["model_answer"],
                result["gold_score"],
                result["model_score"]
                
            )
        return results

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
                evaluation = self.evaluate(prompt, gold_answer, model_answer)

                results.append({
                    "phenomena": phenomena,
                    "item_id": row["item_id"],
                    "prompt": prompt,
                    **evaluation
                })

                total += 1

        logger.info("Total LNRS samples: %d", total)

        for result in results[:5]:
            logger.info(
                "\nItem ID: %s"
                "\nPROMPT:\n%s"
                "\nGOLD ANSWER: %s"
                "\nMODEL ANSWER: %s"
                "\nGOLD SCORE: %f"
                "\nMODEL SCORE: %f"
                "\n--------------------------------",
                result["item_id"],
                result["prompt"],
                result["gold_answer"],
                result["model_answer"],
                result["gold_score"],
                result["model_score"]
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
            evaluation = self.evaluate(prompt, gold_answer, model_answer)
            results.append({
                "prompt": prompt,
                **evaluation
            })
            total +=1

        logger.info("Total Social_IQA samples: %d", total)
        
        for result in results[:5]:
            logger.info(
                "\nPROMPT:\n%s"
                "\nGOLD ANSWER: %s"
                "\nMODEL ANSWER: %s"
                "\nGOLD SCORE: %f"
                "\nMODEL SCORE: %f"
                "\n--------------------------------",
                result["prompt"],
                result["gold_answer"],
                result["model_answer"],
                result["gold_score"],
                result["model_score"]
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
    evaluator = LNRSEvaluator(model_name=args.model_name, judge_model_name="Qwen/Qwen2.5-32B-Instruct")
    evaluator.load_model()
    evaluator.load_judge_model()

    if args.dataset_name == "UCL-DARK/ludwig":
        logger.info("LNRS Evaluation on the ludwig dataset")
        results =evaluator.ludwig_evaluate(dataset["test"])
    elif args.dataset_name == "lm-pragmatics":
        logger.info("LNRS Evaluation on the Pragmega dataset")
        results = evaluator.pragmega_evaluate(dataset)
    elif args.dataset_name == "allenai/social_i_qa":
        logger.info("LNRS Evaluation on the Social-IQA dataset")
        results = evaluator.social_iqa_evaluate(dataset["validation"])
        
    lnrs = evaluator.calculate_lnrs(results)
    logger.info("LNRS Score: %f", lnrs)