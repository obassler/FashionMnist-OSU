import yaml
import random
import os
from train_multiple import setup_logging, train_model


def load_sweep_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def sample_from_config(sweep_config):
    sampled_config = {}
    for key, value in sweep_config.items():
        if isinstance(value, dict):
            sampled_config[key] = {}
            for param, param_value in value.items():
                if isinstance(param_value, list) and len(param_value) == 2 and all(isinstance(v, (int, float)) for v in param_value):
                    sampled_config[key][param] = round(random.uniform(*param_value), 6)
                else:
                    sampled_config[key][param] = param_value
        else:
            sampled_config[key] = value
    return sampled_config

def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "sweep_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    sweep_config = load_sweep_config(config_path)
    num_runs = sweep_config.get("training_cycles", 5)
    base_log_dir = sweep_config.get("paths", {}).get("log_dir", "logs/")  # Default: "logs/"

    setup_logging(base_log_dir)

    results = []
    for i in range(num_runs):
        sampled_config = sample_from_config(sweep_config)
        run_name = f"run_{i + 1}"
        print(f"\n===== Training {run_name} =====")

        val_acc = train_model(sampled_config, run_name=run_name)
        results.append((run_name, val_acc))
        print(f"{run_name} finished with val_acc = {val_acc:.4f}")

    print("\n==== SUMMARY ====")
    for run_name, acc in results:
        print(f"{run_name}: {acc:.4f}")


if __name__ == "__main__":
    main()