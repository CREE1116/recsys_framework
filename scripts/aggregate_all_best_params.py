import os
import json

def aggregate_all_baselines(output_dir_base="output/paper_baselines"):
    if not os.path.exists(output_dir_base):
        print(f"Directory {output_dir_base} does not exist.")
        return

    for dataset_name in os.listdir(output_dir_base):
        dataset_dir = os.path.join(output_dir_base, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
            
        agg_file = os.path.join(dataset_dir, f"best_hyperparameters_{dataset_name}.json")
        print(f"\nScanning and aggregating ALL best hyperparameters for {dataset_name} to {agg_file}...")
        
        all_results = {}
        for sub_dir in os.listdir(dataset_dir):
            sub_dir_path = os.path.join(dataset_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                for file in os.listdir(sub_dir_path):
                    if file.startswith(f"result_{dataset_name}_") and file.endswith(".json"):
                        try:
                            with open(os.path.join(sub_dir_path, file), 'r', encoding='utf-8') as f:
                                res = json.load(f)
                                model_name = res.get('model', 'unknown')
                                
                                # Overwrite best_metric with final test metric
                                best_dir = res.get('best_dir')
                                if best_dir and os.path.exists(os.path.join(best_dir, "final_metrics.json")):
                                    with open(os.path.join(best_dir, "final_metrics.json"), 'r', encoding='utf-8') as fm:
                                        final_metrics = json.load(fm)
                                        res['val_metric'] = res.get('best_metric', None)
                                        # Use standard metric from JSON, fallback to NDCG@10
                                        res['best_metric'] = final_metrics.get('NDCG@10', res['val_metric'])
                                        res['final_test_metrics'] = final_metrics
                                        
                                all_results[model_name] = res
                        except Exception as e:
                            print(f"Error reading {file}: {e}")
        
        if all_results:
            os.makedirs(os.path.dirname(agg_file), exist_ok=True)
            with open(agg_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4)
            print(f"Successfully aggregated {len(all_results)} models for {dataset_name}.")
        else:
            print(f"No results found for {dataset_name}.")

if __name__ == "__main__":
    aggregate_all_baselines()
