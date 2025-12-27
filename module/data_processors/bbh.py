import os
import re
from datasets import load_dataset, DatasetDict, concatenate_datasets
from module.data import DataProcessor, register_processor

@register_processor('bbh')
class BBHDataProcessor(DataProcessor):

    def __init__(self, data_dir):
        super().__init__(data_dir)

    def load_from_raw_dataset(self):
        shot_dataset = load_dataset("Joschka/big_bench_hard", "few_shot_prompts", split="few_shot_prompts")
        
        task_prompts = {ex['dataset_name']: ex['chain_of_thought_prompt'] for ex in shot_dataset}

        tasks = [ex['dataset_name'] for ex in shot_dataset]

        def process_task_data(task_name):
            ds = load_dataset("Joschka/big_bench_hard", task_name, split=task_name)
            prefix = task_prompts.get(task_name, "")

            def convert(example):
                ground_truth = example["target"].strip()
                prompt = f"{prefix}\n\nQUESTION: {example['question']}\n"
                if "choices" in example:
                    prompt += "OPTIONS:\n"
                    for opt, txt in zip(example['choices']['label'], example['choices']['text']):
                        prompt += f"{opt}: {txt}\n"
                elif "OPTIONS" in prefix:
                    prompt += prefix[prefix.index("OPTIONS:"):prefix.index["Let's"]].strip() + "\n"
                prompt += "Let's think step by step.\n"
                
                return {
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "task": task_name
                }

            return ds.map(convert, remove_columns=ds.column_names)

        dataset_list = [process_task_data(task) for task in tasks]
        test_list = [d.select(range(-50, None)) for d in dataset_list]
        train_list = [d.select(range(0, len(d) - 50)) for d in dataset_list]
        self.dataset = DatasetDict({
            "train": concatenate_datasets(train_list),
            "test": concatenate_datasets(test_list)
        })

        self.config['name'] = 'bbh'
        self.config['evaluate_approach'] = 'exact-match'

    def _extract_answer(self, model_output):
        text = model_output.strip()
        match = re.search(r"(?:ANSWER:?|The answer is)\s*(.*)", text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
        else:
            lines = [l for l in text.split('\n') if l.strip()]
            extracted = lines[-1] if lines else ""

        extracted = re.sub(r"[\(\)\.]", "", extracted).strip()
        return extracted

    def _check_answer(self, model_answer, ground_truth):
        model_answer = model_answer.strip()
        extracted = self._extract_answer(model_answer)
        gt_clean = re.sub(r"[\(\)\.]", "", ground_truth).strip()
        return extracted.lower() == gt_clean.lower()

    def evaluate_answer(self, batch_size=16, **kwargs):
        for split in self.dataset.keys():
            def judge(batch):
                results = [1.0 if self._check_answer(m, g) else 0.0 
                           for m, g in zip(batch['model_answer'], batch['ground_truth'])]
                return {"label": results}
            
            self.dataset[split] = self.dataset[split].map(judge, batched=True, batch_size=batch_size)