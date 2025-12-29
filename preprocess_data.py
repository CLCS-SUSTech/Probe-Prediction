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
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'test'], help='Dataset splits to process.')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.work_dir, "datasets"), exist_ok=True)

    for dataset in args.datasets:
        dataset = dataset.lower()
        stage1_dir = os.path.join(args.work_dir, "datasets", "stage1", dataset)
        stage2_dir = os.path.join(args.work_dir, "datasets", args.model_name.replace("/", "--"), args.revision, dataset, "stage2")
        stage3_dir = os.path.join(args.work_dir, "datasets", args.model_name.replace("/", "--"), args.revision, dataset, "stage3")
        processor = None
        flag = True
        for split in args.splits:
            split_dir = os.path.join(stage3_dir, split)
            flag = flag and os.path.exists(split_dir) and len(os.listdir(split_dir)) > 0
        if not flag:
            print(f"Some splits from process stage 3 for dataset {dataset} not found, preprocessing...")
            flag2 = True
            for split in args.splits:
                split_dir = os.path.join(stage2_dir, split)
                flag2 = flag2 and os.path.exists(split_dir) and len(os.listdir(split_dir)) > 0
            if not flag2:
                print(f"Some splits from process stage 2 for dataset {dataset} not found, preprocessing...")
                if not os.path.exists(stage1_dir):
                    print(f"Process Stage 1 for dataset {dataset} not found, preprocessing...")
                processor = load_dataset_processor(dataset, stage1_dir)
                processor.load()
                to_process_splits = []
                for split in args.splits:
                    split_dir = os.path.join(stage2_dir, split)
                    if not os.path.exists(split_dir) or len(os.listdir(split_dir)) == 0:
                        to_process_splits.append(split)
                processor.generate_answer(model=args.model_name, revision=args.revision, tensor_parallel_size=args.tensor_parallel_size, batch_size=args.batch_size, splits=to_process_splits)
                processor.save(stage2_dir)
                print(f"Stage 2 processing for dataset {dataset} completed and saved to {stage2_dir}.")
            if processor is None:
                processor = load_dataset_processor(dataset, stage2_dir)
                processor.load()
            to_process_splits = []
            for split in args.splits:
                split_dir = os.path.join(stage3_dir, split)
                if not os.path.exists(split_dir) or len(os.listdir(split_dir)) == 0:
                    to_process_splits.append(split)
            processor.evaluate_answer(tensor_parallel_size=args.tensor_parallel_size, batch_size=args.batch_size, splits=to_process_splits)
            processor.save(stage3_dir)
            print(f"Stage 3 processing for dataset {dataset} completed and saved to {stage3_dir}.")
        else:
            print(f"Stage 3 for dataset {dataset} already exists at {stage3_dir}, skipping processing.")