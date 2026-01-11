
from typing import List
from utils import pass_k
from vllm import LLM, SamplingParams
from abc import ABC, abstractmethod
import os
import json
from datasets import load_from_disk, Dataset
import warnings
import math
from typing import Callable

class DataProcessor(ABC):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.config = {
            'name': None,
            'model': None,
            'revision': None,
            'evaluate_approach': None,
        }
        self.dataset = None

    @abstractmethod
    def load_from_raw_dataset(self):
        raise NotImplementedError

    def get(self, split="train"):
        return self.dataset[split]
    
    def _save_config(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        with open(os.path.join(self.data_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def _load_config(self):
        assert os.path.exists(self.data_dir), f"Data directory {self.data_dir} does not exist."
        assert os.path.exists(os.path.join(self.data_dir, 'config.json')), f"Config file does not exist in {self.data_dir}."
        with open(os.path.join(self.data_dir, 'config.json'), 'r') as f:
            self.config = json.load(f)
    
    def save(self, new_data_dir):
        if not os.path.exists(new_data_dir):
            os.makedirs(new_data_dir)
        with open(os.path.join(new_data_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        self.dataset.save_to_disk(new_data_dir)
        
    def load(self):
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist. Loading from raw dataset.")
            self.load_from_raw_dataset()          
            print(f"Loaded from raw dataset. Saving processed dataset to {self.data_dir}.")
            self.save(self.data_dir)
        else:
            print(f"Loading processed dataset from {self.data_dir}.")
            if os.path.exists(os.path.join(self.data_dir, 'config.json')):
                with open(os.path.join(self.data_dir, 'config.json'), 'r') as f:
                    self.config = json.load(f)
            self.dataset = load_from_disk(self.data_dir)

        print(f"Dataset Information:")
        print(json.dumps(self.config, indent=4))
    
    @property
    def name(self):
        return self.config['name']
    
    @property
    def model(self):
        return self.config['model']
    
    @property
    def revision(self):
        return self.config['revision']
    
    @property
    def evaluate_approach(self):
        return self.config['evaluate_approach']
    
    def generate_answer(self,
                        model,
                        revision="main",
                        tensor_parallel_size=1,
                        temperature=0.0,
                        max_tokens=1024,
                        top_p=1,
                        n=1,
                        stop=["\n\nQuestion", "\n\nQ:"],
                        splits=["train", "test"]):
        assert self.dataset is not None, "Dataset is not loaded. Please load the dataset before generating answers."

        llm = LLM(model=model,
                  revision=revision,
                  tensor_parallel_size=tensor_parallel_size)
        sampling_params = SamplingParams(temperature=temperature,
                                         max_tokens=max_tokens,
                                         top_p=top_p,
                                         stop=stop)
        
        all_prompts = []
        prompt_counts_per_split = []
        for split in splits:
            prompts = self.dataset[split]['prompt']
            all_prompts.extend([p for p in prompts for _ in range(n)])
            prompt_counts_per_split.append(len(prompts) * n)

        all_responses = llm.generate(all_prompts, sampling_params)
        all_responses = [response.outputs[0].text for response in all_responses]

        idx = 0
        for split, count in zip(splits, prompt_counts_per_split):
            responses_for_split = all_responses[idx:idx + count]
            responses_per_prompt = []
            for i in range(0, len(responses_for_split), n):
                responses_per_prompt.append(responses_for_split[i:i + n])
            self.dataset[split] = self.dataset[split].add_column('model_answer', responses_per_prompt)
            idx += count

        self.config['model'] = model
        self.config['revision'] = revision
            
    @abstractmethod
    def evaluate_answer(self, **kwargs):
        pass

    @staticmethod
    def label_balance(dataset: Dataset, label_column: str, positive_label_value, limit: int) -> list:
        total_size = len(dataset)

        effective_limit = min(limit, total_size)
        assert limit > 0, "Limit must be a positive integer."
        if total_size < limit:
            warnings.warn(
                f"The provided limit ({limit}) exceeds the dataset size ({total_size}). The entire dataset will be used instead."
            )

        positive_indices = [i for i, label in enumerate(dataset[label_column]) if label == positive_label_value]
        negative_indices = [i for i, label in enumerate(dataset[label_column]) if label != positive_label_value]

        num_pos = len(positive_indices)
        num_neg = len(negative_indices)

        if num_pos >= num_neg:
            larger_indices = positive_indices
            smaller_indices = negative_indices
            num_larger = num_pos
            num_smaller = num_neg
        else:
            larger_indices = negative_indices
            smaller_indices = positive_indices
            num_larger = num_neg
            num_smaller = num_pos

        target_half = math.floor(effective_limit / 2)

        if num_pos >= target_half and num_neg >= target_half:
            num_take_smaller = target_half
            num_take_larger = effective_limit - num_take_smaller
            # Notice: for odd effective_limit, larger class gets the extra one
        else:
            num_take_smaller = num_smaller
            remaining_to_take = effective_limit - num_take_smaller
            
            num_take_larger = min(num_larger, remaining_to_take)
        
        selected_smaller_indices = smaller_indices[:num_take_smaller]
        selected_larger_indices = larger_indices[:num_take_larger]

        selected_indices = selected_smaller_indices + selected_larger_indices
            
        return selected_indices
    
    def _cal_passk(self, model_answer: List[str], ground_truth: str, k: int, _check_answer: Callable=None) -> float:
        if _check_answer is None:
            c = sum([self._check_answer(ans, ground_truth) for ans in model_answer])
        else:
            c = sum([_check_answer(ans, ground_truth) for ans in model_answer])
        n = len(model_answer)
        return pass_k(c, k, n)

class ProcessorRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name=None):
        def decorator(cls):
            key = name or cls.__name__.lower()
            if key in self._registry:
                raise ValueError(f"Processor '{key}' already registered.")
            self._registry[key] = cls
            return cls
        return decorator

    def get(self, name):
        try:
            return self._registry[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown processor: {name}")

    def list(self):
        return list(self._registry.keys())

registry = ProcessorRegistry()
register_processor = registry.register
def _load_processor_cls():
    import importlib
    import pkgutil
    import module.data_processors
    package = module.data_processors
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package.__name__}.{module_name}")

_load_processor_cls()

print(f"Registered dataset processors: {registry.list()}")

def load_dataset_processor(name: str, data_dir : str) -> DataProcessor:
    '''
    Arguments:
        - name: the name of the dataset processor
        - data_dir: the directory to load/save the processed dataset
    
    Returns:
        - An instance of DataProcessor for the specified dataset.

    Dataset columns description:
      - prompt: few-shot prompts with chain of thought and problem description
      - ground_truth: the correct answer
      - model_answer: the inference answer from the model
      - label: the correctness of model predictions (1 for correct, 0 for incorrect)

    Processor config:
      - name: the name of the dataset
      - model: the model used for generating answers
      - revision: the model revision used for generating answers
      - evaluate_approach: the answer format of the benchmark (exact-match, llm-as-judge)
    
    Currently supported datasets:
        - 'math': MATH dataset
        - 'aime': AIME dataset
    '''
    if name not in registry.list():
        raise ValueError(f"Unknown dataset: {name}")
    processor_cls = registry.get(name)
    processor = processor_cls(data_dir)
    return processor