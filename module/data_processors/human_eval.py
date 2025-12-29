import os
import random
from module.data import DataProcessor, register_processor
from datasets import load_dataset, DatasetDict
import evaluate as hf_evaluate

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
    
    def generate_answer(self,
                        model,
                        revision="main",
                        tensor_parallel_size=1,
                        temperature=0.0,
                        max_tokens=1024,
                        top_p=1,
                        stop=["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
                        splits=["train", "test"],
                        **kwargs):
        default_stop = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
        if stop != default_stop:
            print(f"Warning: The provided stop tokens {stop} differ from the default stop tokens {default_stop}. To ensure proper code generation, the settings will be changed back to the default stop tokens.")
        super().generate_answer(
            model,
            revision,
            tensor_parallel_size,
            temperature,
            max_tokens,
            top_p,
            stop=default_stop,
            splits=splits,
            **kwargs
        )
    
    def evaluate_answer(self, batch_size=16, splits=["train", "test"], **kwargs):
        for split in splits:
            assert self.dataset is not None and "model_answer" in self.dataset[split].column_names, f"Model answers not found in the dataset split '{split}'. Please generate model answers before evaluation."
            dataset = self.dataset[split]
            def judge(batch):
                
                predictions = [[p + (ans if ans.find("```") == -1 else ans[: ans.find("```")])] for p, ans in zip(batch["prompt"], batch["model_answer"])]
                
                references = [
                    f"{test_code}\ncheck({entry_point})" 
                    for test_code, entry_point in zip(batch["test"], batch["entry_point"])
                ]
                results, logs = self.code_eval.compute(
                    predictions=predictions, 
                    references=references, 
                    k=[1]
                )
                labels = []
                for i in range(len(batch["prompt"])):
                    labels.append(1.0 if logs[i][0][1]['passed'] else 0.0) # only work with num_process=1

                return {
                    'label': labels
                }
            
            judge_dataset = dataset.map(judge, batched=True, batch_size=batch_size)
            self.dataset[split] = judge_dataset