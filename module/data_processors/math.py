from vllm import LLM, SamplingParams
import os
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from module.data import DataProcessor, register_processor
import json

@register_processor('math')
class MathDataProcessor(DataProcessor):

    def __init__(self, data_dir):
        super().__init__(data_dir)

    def load_from_raw_dataset(self):
        trainset = load_dataset("qwedsacf/competition_math", split="train")
        fewshots = trainset.select(range(4))
        trainset = trainset.select(range(4, len(trainset)))
        testset = load_dataset("HuggingFaceH4/MATH-500", split="test")

        def format_fewshot(example):
            return f"Question: {example['problem']}\n\nAnswer: {example['solution']}"

        few_shot_examples = [format_fewshot(ex) for ex in fewshots]
        few_shot_block = "\n\n".join(few_shot_examples)

        def convert(example):
            prompt = (
                "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
                f"{few_shot_block}\n\n"
                f"Question: {example['problem']}\n\nAnswer:"
            )

            return {
                "prompt": prompt,
                "question": example["problem"],
                "ground_truth": example["solution"],
            }

        trainset = trainset.map(convert, remove_columns=trainset.column_names)
        testset = testset.map(convert, remove_columns=testset.column_names)

        def remove_overlap(testset, trainset):
            test_questions = set(testset["question"])
            return trainset.filter(lambda x: x["question"] not in test_questions)

        trainset = remove_overlap(testset, trainset)

        self.dataset = DatasetDict({
            "train": trainset,
            "test": testset
        })
        
        self.config['name'] = 'math'
        self.config['evaluate_approach'] = 'llm-as-judge'
    
    def evaluate_answer(self, model = "Qwen/Qwen3-30B-A3B-Instruct-2507", tensor_parallel_size=1, batch_size=16, splits=["train", "test"], **kwargs):
        prompt = """You are required to help with a math problem evaluation. \
Given an input text and a groundtruth answer, your task is to judge \
whether the final answer given by the input is correct or not. \
If the answer is correct, return 'True', otherwise, return 'False'. \

Input:
Weng earns $12 an hour. Yesterday, she worked 50 minutes. So she actually worked for 50 minutes / 60 hours = 5/6 hours. \
The pay for 5/6 hours is 5/6 * $12 = $10. So the answer is $10.

Groundtruth:
10

Output:
True

Input:
Normally, a single ship needs 847 people. In 3 centuries, there should be 3 * 847 = 2541 people that are eaten. \
Since they each have twice as many people, there are 2541 / 2 = 1270. Since monster kills a ship every century, \
and this was the first, it means it ate 1270 (left) - 847 (already eaten) = 423 people.

Groundtruth:
121

Output:
False

Input:
Let\u2019s calculate the total before discount. $15.00 x 2 is $30.00 $3.00 x 6 is $18.00 $2.00 x 6 is $12.00 Total: \
$30.00 + $18.00 + $12.00 = $60.00 So Silvia\u2019s order will be 10% off $60.00. 10% of total amount is $6.00. \
$60.00 \u2013 $6.00 = $54.00. So the answer is $54.00.",

Groundtruth:
54

Output:
True

Input:
{}

Groundtruth:
{}

Output:
"""
        llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size)

        splits = self.dataset.keys()
        for split in splits:
            assert self.dataset is not None and "model_answer" in self.dataset[split].column_names, f"Model answers not found in the dataset split '{split}'. Please generate model answers before evaluation."
            dataset = self.dataset[split]
            sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
            inputs = []
            for model_answer, ground_truth in zip(dataset["model_answer"], dataset["ground_truth"]
            ):
                input_text = prompt.format(model_answer.strip(), ground_truth)
                inputs.append(input_text)
            outputs = llm.generate(inputs, sampling_params)
            outputs = [output.outputs[0].text.strip() for output in outputs]
            results = [output.lower().startswith("true") or output.lower().startswith("yes") for output in outputs]
            results = [1.0 if res else 0.0 for res in results]
            self.dataset[split] = dataset.add_column("label", results)