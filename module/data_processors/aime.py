import os
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from module.data import DataProcessor, register_processor
import json
import re

@register_processor('aime')
class AimeDataProcessor(DataProcessor):

    def __init__(self, data_dir):
        super().__init__(data_dir)

    def load_from_raw_dataset(self):
        instruction = "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        def convert(example):
            prompt_text = example.get("Question") or example.get("problem")
            prompt = instruction + prompt_text

            answer = example.get("Answer") or example.get("answer") or ""

            return {
                "prompt": prompt,
                "ground_truth": answer,
            }

        trainset = load_dataset("gneubig/aime-1983-2024")['train']
        trainset = trainset.map(convert, remove_columns=trainset.column_names)
        testset = [
            load_dataset("HuggingFaceH4/aime_2024", split="train"),
            load_dataset("math-ai/aime25", split="test")
        ]
        testset = concatenate_datasets([
            ts.map(convert, remove_columns=ts.column_names) for ts in testset
        ])

        self.dataset = DatasetDict({
            "train": trainset,
            "test": testset
        })
        
        self.config['name'] = 'aime'
        self.config['evaluate_approach'] = 'exact-match'
    
    def _check_answer(self, model_answer, ground_truth):
        model_answer = model_answer.strip()
        ground_truth = ground_truth.strip()
        if "\\boxed{" in model_answer:
            ans_text = model_answer.split('\\boxed{')[-1]
            extracted_answer = ans_text.split('}')[0].strip()
            return extracted_answer == ground_truth
        return False

    def evaluate_answer(self, batch_size=16, splits=["train", "test"], k=1, **kwargs):
        splits = self.dataset.keys()
        for split in splits:
            assert self.dataset is not None and "model_answer" in self.dataset[split].column_names, f"Model answers not found in the dataset split '{split}'. Please generate model answers before evaluation."
            
            def judge(batch):
                results = [self._cal_passk(m, g, k) # when n=1 and k=1, pass@1(n=1) is equivalent to exact match accuracy
                           for m, g in zip(batch['model_answer'], batch['ground_truth'])] # `model_answer` should be a list in the new ver. of llm generation
                return {"label": results}

            self.dataset[split] = self.dataset[split].map(judge, batched=True, batch_size=batch_size)