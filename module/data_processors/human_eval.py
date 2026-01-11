import os
import random
from module.data import DataProcessor, register_processor
from datasets import load_dataset, DatasetDict
import evaluate as hf_evaluate
from utils import pass_k

@register_processor('human_eval')
class HumanEvalDataProcessor(DataProcessor):

    def __init__(self, data_dir, train_ratio=0.61, seed=42):
        super().__init__(data_dir)
        self.seed = seed
        self.train_ratio = train_ratio
        try:
            self.code_eval = hf_evaluate.load("code_eval")
        except Exception as e:
            print(f"Error loading code_eval: {e}. Make sure to set os.environ['HF_ALLOW_CODE_EVAL'] = '1'")

    def load_from_raw_dataset(self):
        random.seed(self.seed)
        dataset = load_dataset("openai/openai_humaneval")["test"]
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        train_len = int(len(dataset) * self.train_ratio)
        train_indices = indices[:train_len]
        test_indices = indices[train_len:]

        train_raw = dataset.select(train_indices)
        test_raw = dataset.select(test_indices)

        def convert(example):
            return {
                "prompt": example["prompt"],
                "ground_truth": example["canonical_solution"],
                "task_id": example["task_id"],
                "test": example["test"],
                "entry_point": example["entry_point"],
            }

        trainset = train_raw.map(convert, remove_columns=train_raw.column_names)
        testset = test_raw.map(convert, remove_columns=test_raw.column_names)
        self.dataset = DatasetDict({
            "train": trainset,
            "test": testset
        })

        self.config['name'] = 'human_eval'
        self.config['evaluate_approach'] = 'code-execution'
        
    def evaluate_answer(self, batch_size=16, splits=["train", "test"], k=1, **kwargs):
        for split in splits:
            assert self.dataset is not None and "model_answer" in self.dataset[split].column_names, f"Model answers not found in the dataset split '{split}'. Please generate model answers before evaluation."
            dataset = self.dataset[split]
            def judge(batch):

                predictions = []
                for prompt, answers in zip(batch["prompt"], batch["model_answer"]):
                    sample_preds = []
                    for ans in answers:
                        clean_ans = ans
                        if "```" in ans:
                            clean_ans = ans[:ans.find("```")]
                        sample_preds.append(prompt + "\n" + clean_ans)
                    predictions.append(sample_preds)

                references = []
                for test_code, entry_point in zip(batch["test"], batch["entry_point"]):
                    references.append(f"{test_code}\ncheck({entry_point})")
                
                results, logs = self.code_eval.compute(
                    predictions=predictions, 
                    references=references, 
                    k=[1]
                )

                labels = []
                for i in range(len(batch["prompt"])):
                    problem_logs = logs[i]
                    n = len(problem_logs)
                    
                    c = 0
                    for _, result in problem_logs:
                        if result.get('passed', False):
                            c += 1
                            
                    score = pass_k(c, k, n)
                    labels.append(score)

                return {
                    'label': labels
                }
            
            judge_dataset = dataset.map(judge, batched=True, batch_size=batch_size)
            self.dataset[split] = judge_dataset