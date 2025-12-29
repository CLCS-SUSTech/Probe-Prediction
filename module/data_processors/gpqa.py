import os
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from module.data import DataProcessor, register_processor
import json
import random

@register_processor('gpqa')
class GPQADataProcessor(DataProcessor):

    def __init__(self, data_dir, train_ratio=0.782, seed=42):
        super().__init__(data_dir)
        self.seed = seed
        self.train_ratio = train_ratio

    def load_from_raw_dataset(self):

        random.seed(self.seed)

        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")['train']

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        train_len = int(len(indices) * self.train_ratio)
        train_indices = indices[:train_len]
        test_indices = indices[train_len:]

        train_raw = dataset.select(train_indices)
        test_raw = dataset.select(test_indices)

        instruction = "Please reason step by step, and put your final answer within \\boxed{}.\n\n"

        def convert(example):
            question_text = example.get("Question")
            correct = example.get("Correct Answer")
            incorrects = [
                example.get("Incorrect Answer 1"),
                example.get("Incorrect Answer 2"),
                example.get("Incorrect Answer 3")
            ]

            choices = incorrects + [correct]
            random.shuffle(choices)

            correct_index = choices.index(correct)
            letters = ["A", "B", "C", "D"]
            answer_letter = letters[correct_index]

            choices_text = "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices)])
            prompt = instruction + "Qusetion:\n" + question_text + "\n" + choices_text + "\n\nAnswer:"

            return {
                "prompt": prompt,
                "ground_truth": answer_letter,
            }

        trainset = train_raw.map(convert, remove_columns=train_raw.column_names)
        testset = test_raw.map(convert, remove_columns=train_raw.column_names)

        self.dataset = DatasetDict({
            "train": trainset,
            "test": testset
        })
        
        self.config['name'] = 'gpqa'
        self.config['evaluate_approach'] = 'exact-match'

    def _check_answer(self, model_answer : str, ground_truth : str):
        model_answer = model_answer.strip()
        if "\\boxed{" in model_answer:
            ans_text = model_answer.split('\\boxed{')[-1]
            extracted_answer = ans_text.split('}')[0].strip()
            return extracted_answer.lower().startswith(ground_truth.lower())
        return False

    def evaluate_answer(self, batch_size=16, splits=["train", "test"], **kwargs):
        for split in splits:
            assert self.dataset is not None and "model_answer" in self.dataset[split].column_names,  f"Model answers not found in the dataset '{split}'. Please generate model answers before evaluation."
            dataset = self.dataset[split]
            
            def judge(batch):
                results = [1.0 if self._check_answer(m, g) else 0.0 
                           for m, g in zip(batch['model_answer'], batch['ground_truth'])]
                return {"label": results}

            judged_dataset = dataset.map(judge, batched=True, batch_size=batch_size)
            self.dataset[split] = judged_dataset