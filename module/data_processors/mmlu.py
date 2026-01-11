import os
import re
from datasets import load_dataset, DatasetDict
from module.data import DataProcessor, register_processor

@register_processor('mmlu')
class MMLUDataProcessor(DataProcessor):

    def __init__(self, data_dir):
        super().__init__(data_dir)

    def load_from_raw_dataset(self, train_limit=10000):
        dataset = load_dataset("bin957/mmlu_processed")
        trainset = dataset['train']
        data_names = list(set(trainset['data_name']))
        avg = train_limit // len(data_names)
        selected_indices = []
        for name in data_names:
            name_indices = [i for i, n in enumerate(trainset['data_name']) if n == name]
            selected_name_indices = name_indices[:avg]
            selected_indices.extend(selected_name_indices)
        trainset = trainset.select(selected_indices)
        testset = dataset['test']
        self.dataset = DatasetDict({
            'train': trainset,
            'test': testset
        })

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

    def evaluate_answer(self, batch_size=16, splits=["train", "test"], num_proc=8, k=1, **kwargs):
        for split in splits:
            assert self.dataset is not None and "model_answer" in self.dataset[split].column_names, "Model answers not found. Please generate model answers before evaluation."
            dataset:DatasetDict = self.dataset[split]
            
            def judge(batch):
                results = [self._cal_passk(m, g, k) # when n=1 and k=1, pass@1(n=1) is equivalent to exact match accuracy
                           for m, g in zip(batch['model_answer'], batch['ground_truth'])] # `model_answer` should be a list in the new ver. of llm generation
                return {"label": results}

            judged_dataset = dataset.map(judge, batched=True, batch_size=batch_size, num_proc=num_proc)
            self.dataset[split] = judged_dataset