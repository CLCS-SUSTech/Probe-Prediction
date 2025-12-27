import os
import re
from datasets import load_dataset, DatasetDict, concatenate_datasets
from module.data import DataProcessor, register_processor
import evaluate as hf_evaluate

@register_processor('mbpp')
class MBPPDataProcessor(DataProcessor):

    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.code_eval = hf_evaluate.load("code_eval")
        try:
            self.code_eval = hf_evaluate.load("code_eval")
        except Exception as e:
            print(f"Error loading code_eval: {e}. Make sure to set os.environ['HF_ALLOW_CODE_EVAL'] = '1'")

    def load_from_raw_dataset(self):
        raw_dataset = load_dataset("google-research-datasets/mbpp", "full")
        
        test_full = raw_dataset['test']
        test_for_train = test_full.select(range(250))
        test_for_eval = test_full.select(range(250, 500))

        fewshot_samples = raw_dataset['train'].select(range(3))
        
        def format_doc(example, is_fewshot=False):
            tests = example["test_list"][:3]
            tests_str = "\n".join(tests)
            
            text = (
                f"You are an expert Python programmer, and here is your task: {example['text']} "
                f"Your code should pass these tests:\n\n{tests_str}\n[BEGIN]\n"
            )
            
            if is_fewshot:
                text += f"{example['code']}\n[DONE]\n"
            return text

        fewshot_prefix = "".join([format_doc(ex, is_fewshot=True) for ex in fewshot_samples])
        fewshot_questions = set(fewshot_samples['text'])

        def convert(example):
            prompt = fewshot_prefix + format_doc(example, is_fewshot=False)
            
            return {
                "prompt": prompt,
                "ground_truth": "\n".join(example["test_list"]),
                "tests": example["test_list"],
                "_is_shot": example["text"] in fewshot_questions
            }

        train_combined = concatenate_datasets([raw_dataset['train'], raw_dataset['validation'], test_for_train])
        
        self.dataset = DatasetDict({
            "train": train_combined.map(convert).filter(lambda x: not x["_is_shot"]),
            "test": test_for_eval.map(convert).filter(lambda x: not x["_is_shot"])
        })

        for split in self.dataset:
            self.dataset[split] = self.dataset[split].remove_columns(["_is_shot"])

        self.config['name'] = 'mbpp'
        self.config['evaluate_approach'] = 'code-execution'

    def _extract_answer(self, model_output):
        code = model_output
        if "[DONE]" in code:
            code = code.split("[DONE]")[0]
        
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
            
        return code.strip()
    
    def generate_answer(self,
                        model,
                        revision="main",
                        tensor_parallel_size=1,
                        temperature=0.0,
                        max_tokens=1024,
                        top_p=1,
                        stop=["[DONE]"],
                        batch_size=16):
        default_stop = ["[DONE]"]
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
            batch_size=batch_size
        )

    def evaluate_answer(self, batch_size=16, **kwargs):
        assert self.dataset is not None and "model_answer" in self.dataset[list(self.dataset.keys())[0]].column_names, \
            "Model answers not found. Please generate model answers before evaluation."

        splits = self.dataset.keys()

        for split in splits:
            dataset = self.dataset[split]

            def judge(batch):
                def clean_gen(ans):
                    if "[DONE]" in ans:
                        ans = ans.split("[DONE]")[0]
                    if ans.find("```") != -1:
                        ans = ans[:ans.find("```")]
                    return ans

                predictions = [[p + clean_gen(ans)] for p, ans in zip(batch["prompt"], batch["model_answer"])]
                references = [test_code for test_code in batch["tests"]]
                
                results, logs = self.code_eval.compute(
                    predictions=predictions, 
                    references=references, 
                    k=[1]
                )
                
                labels = []
                for i in range(len(batch["prompt"])):
                    labels.append(1.0 if logs[i][0][1]['passed'] else 0.0)

                return {
                    'label': labels
                }
            
            judge_dataset = dataset.map(judge, batched=True, batch_size=batch_size)
            self.dataset[split] = judge_dataset