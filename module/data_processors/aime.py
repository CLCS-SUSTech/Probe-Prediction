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
    
    def check_answer(self, model_answer, ground_truth):
        model_answer = model_answer.strip()
        ground_truth = ground_truth.strip()
        if "{" in model_answer and "}" in model_answer:
            start = model_answer.index("{") + 1
            end = model_answer.index("}")
            extracted_answer = model_answer[start:end].strip()
            return extracted_answer == ground_truth
        else:
            pattern = r'^\d+'
            match = re.match(pattern, model_answer)
            if match:
                extracted_answer = match.group(0)
                return extracted_answer == ground_truth
            else:
                return False