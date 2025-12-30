import os
import re
from datasets import load_dataset, DatasetDict
from module.data import DataProcessor, register_processor

@register_processor('gsm8k')
class GSM8KDataProcessor(DataProcessor):

    def __init__(self, data_dir):
        super().__init__(data_dir)

    def load_from_raw_dataset(self):

        raw_dataset = load_dataset("openai/gsm8k", "main")

        few_shot_examples = raw_dataset['train'].select(range(8))
        
        few_shot_prefix = ""
        for ex in few_shot_examples:
            few_shot_prefix += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        
        few_shot_q_set = set(few_shot_examples['question'])

        def convert(example):
            return {
                "prompt": few_shot_prefix + "Question: " + example["question"] + "\nAnswer:",
                "ground_truth": example["answer"],
                "_is_shot": example["question"] in few_shot_q_set
            }

        trainset = raw_dataset['train'].map(convert).filter(lambda x: not x["_is_shot"])
        testset = raw_dataset['test'].map(convert).filter(lambda x: not x["_is_shot"])

        self.dataset = DatasetDict({
            "train": trainset.remove_columns(["_is_shot", "question", "answer"]),
            "test": testset.remove_columns(["_is_shot", "question", "answer"])
        })
        
        self.config['name'] = 'gsm8k'
        self.config['evaluate_approach'] = 'exact-match'

    def _extract_answer(self, text):
        strict_match = re.search(r"#### (-?[0-9.,]+)", text)
        if strict_match:
            return strict_match.group(1)

        flexible_pattern = r"(-?[\$0-9.,]{2,})|(-?[0-9]+)"
        matches = re.findall(flexible_pattern, text)
        
        if matches:
            last_match = matches[-1]
            extracted = last_match[0] if last_match[0] else last_match[1]
            return extracted
        
        return ""

    def _check_answer(self, model_answer, ground_truth):
        model_answer = model_answer.strip()
        def clean(text):
            return text.replace(",", "").replace("$", "").strip().rstrip(".")

        extracted_raw = self._extract_answer(model_answer)
        extracted_ground_truth = self._extract_answer(ground_truth)
        
        cleaned_model = clean(extracted_raw)
        cleaned_gt = clean(extracted_ground_truth)

        if not cleaned_model:
            return False

        try:
            return float(cleaned_model) == float(cleaned_gt)
        except:
            return cleaned_model == cleaned_gt

    def evaluate_answer(self, batch_size=16, splits=["train", "test"], k=1, **kwargs):
        for split in splits:
            assert self.dataset is not None and "model_answer" in self.dataset[split].column_names, f"Model answers not found in dataset split '{split}'. Please generate model answers before evaluation."
            dataset = self.dataset[split]
            
            def judge(batch):
                results = [self._cal_passk(m, g, k) # when n=1 and k=1, pass@1(n=1) is equivalent to exact match accuracy
                           for m, g in zip(batch['model_answer'], batch['ground_truth'])] # `model_answer` should be a list in the new ver. of llm generation
                return {"label": results}

            judged_dataset = dataset.map(judge, batched=True, batch_size=batch_size)
            self.dataset[split] = judged_dataset