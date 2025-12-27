import argparse
from module.data import DataProcessor, load_dataset_processor
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True, help='The working directory.')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to preprocess.')
    parser.add_argument('--model_name', type=str, required=True, help='The model name.')
    parser.add_argument('--revision', type=str, default='main', help='The model revision.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor parallel size.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing.')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.work_dir, "datasets"), exist_ok=True)

    for dataset in args.datasets:
        dataset = dataset.lower()
        stage1_dir = os.path.join(args.work_dir, "datasets", "stage1", dataset)
        stage2_dir = os.path.join(args.work_dir, "datasets", args.model_name, dataset, "stage2")
        stage3_dir = os.path.join(args.work_dir, "datasets", args.model_name, dataset, "stage3")
        processor = None
        if not os.path.exists(stage3_dir):
            print(f"Process Stage 3 for dataset {dataset} not found, reprocessing...")
            if not os.path.exists(stage2_dir):
                print(f"Process Stage 2 for dataset {dataset} not found, reprocessing...")
                if not os.path.exists(stage1_dir):
                    print(f"Process Stage 1 for dataset {dataset} not found, reprocessing...")
                processor = load_dataset_processor(dataset, stage1_dir)
                processor.load()
                processor.generate_answer(model=args.model_name, revision=args.revision, tensor_parallel_size=args.tensor_parallel_size, batch_size=args.batch_size)
                processor.save(stage2_dir)
                print(f"Stage 2 processing for dataset {dataset} completed and saved to {stage2_dir}.")
            if processor is None:
                processor = load_dataset_processor(dataset, stage2_dir)
                processor.load()
            processor.evaluate_answer(tensor_parallel_size=args.tensor_parallel_size, batch_size=args.batch_size)
            processor.save(stage3_dir)
            print(f"Stage 3 processing for dataset {dataset} completed and saved to {stage3_dir}.")
        else:
            print(f"Stage 3 for dataset {dataset} already exists at {stage3_dir}, skipping processing.")