import os
import re
from datasets import load_dataset, DatasetDict, concatenate_datasets
from module.data import DataProcessor, register_processor
import evaluate as hf_evaluate
from utils import pass_k

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

    def _extract_code_blocks(self, text: str) -> str:
        pattern = r"```(?:\w+)?\n?(.*?)\n?```"
        matches = re.findall(pattern, r"```" + text, re.DOTALL)
        if not matches:
            text_without_lang = re.sub(r"```python", "```", text)
            matches = re.findall(pattern, text_without_lang, re.DOTALL)
        if not matches:
            return text
        else:
            return matches[0]

    def evaluate_answer(self, batch_size=16, splits=["train", "test"], num_proc=8, k=1, **kwargs):
        for split in splits:
            assert self.dataset is not None and "model_answer" in self.dataset[split].column_names, f"Model answers not found in dataset split '{split}'. Please generate model answers before evaluation."
            dataset = self.dataset[split]

            def judge(batch):
                def clean_gen(ans):
                    if "[DONE]" in ans:
                        ans = ans.split("[DONE]")[0]
                    if ans.find("```") != -1:
                        ans = ans[:ans.find("```")]
                    return ans
                
                predictions = []
                for prompt, answers in zip(batch["prompt"], batch["model_answer"]):
                    sample_preds = []
                    for ans in answers:
                        sample_preds.append(clean_gen(self._extract_code_blocks(ans)))
                    predictions.append(sample_preds)

                references = [
                    "\n".join(tests) for tests in batch["tests"]
                ]
                
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
            
            judge_dataset = dataset.map(judge, batched=True, batch_size=batch_size, num_proc=num_proc)
            self.dataset[split] = judge_dataset