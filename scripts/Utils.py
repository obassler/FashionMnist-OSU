import os
import torch
import re
import json

def export_predictions_per_run(multirun_base_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    run_folder_pattern = re.compile(r'^\d+$')

    for run_dir in os.listdir(multirun_base_path):
        if run_folder_pattern.match(run_dir):
            run_path = os.path.join(multirun_base_path, run_dir)
            if os.path.isdir(run_path):
                run_predictions = {}

                for file_name in os.listdir(run_path):
                    if file_name.startswith('test') and file_name.endswith('.pt'):
                        file_path = os.path.join(run_path, file_name)

                        try:
                            predictions_tensor = torch.load(file_path)

                            if isinstance(predictions_tensor, torch.Tensor):
                                predictions_list = predictions_tensor.cpu().numpy().tolist()
                            else:
                                predictions_list = predictions_tensor

                            run_predictions[file_name] = predictions_list

                            print(f"Loaded predictions from {file_path}")

                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                if run_predictions:
                    output_path = os.path.join(output_dir, f"{run_dir}.json")
                    with open(output_path, "w") as f:
                        json.dump(run_predictions, f, separators=(',', ':'))
                    print(f"Exported run {run_dir} predictions to {output_path}")

    print(f"All runs exported to folder: {output_dir}")


if __name__ == '__main__':
    multirun_path = '/mnt/data/Bassler/FashionMnist-OSU/scripts/multirun/2025-09-28/14-22-44'
    output_dir = 'predictions'
    export_predictions_per_run(multirun_path, output_dir)
