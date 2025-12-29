import os
import re
from datasets import load_dataset, DatasetDict
from module.data import DataProcessor, register_processor

@register_processor('mmlu')
class MMLUDataProcessor(DataProcessor):

    def __init__(self, data_dir):
        super().__init__(data_dir)

    def load_from_raw_dataset(self):
        self.dataset = load_dataset("bin957/mmlu_processed")

        self.config['name'] = 'mmlu'
        self.config['evaluate_approach'] = 'exact-match'

    def _check_answer(self, model_answer, ground_truth):
        model_answer = model_answer.strip()
        pattern = r"The answer is \(([A-D])\)"
        match = re.search(pattern, model_answer)
        
        if match:
            extracted_answer = match.group(1)
        else:
            pattern_backup = r"\(([A-D])\)"
            matches = re.findall(pattern_backup, model_answer)
            if matches:
                extracted_answer = matches[-1]
            else:
                return False

        return extracted_answer.lower() == ground_truth.lower()

    def evaluate_answer(self, batch_size=16, splits=["train", "test"], num_proc=8, **kwargs):
        for split in splits:
            assert self.dataset is not None and "model_answer" in self.dataset[split].column_names, "Model answers not found. Please generate model answers before evaluation."
            dataset:DatasetDict = self.dataset[split]
            
            def judge(batch):
                results = []
                for model_answer, ground_truth in zip(batch['model_answer'], batch['ground_truth']):
                    is_correct = self._check_answer(model_answer, ground_truth)
                    results.append(1.0 if is_correct else 0.0)
                return {"label": results}

            judged_dataset = dataset.map(judge, batched=True, batch_size=batch_size, num_proc=num_proc)
            self.dataset[split] = judged_dataset