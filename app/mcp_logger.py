import json
import os
from datetime import datetime

def log_model_metadata(model_dir, metrics):
    metadata = {
        "model_name": "bert-base-uncased",
        "version": datetime.now().strftime("%Y%m%d%H%M%S"),
        "date": datetime.now().isoformat(),
        "framework": "PyTorch",
        "task": "Sentiment Classification",
        "metrics": metrics,
        "author": "Your Name",
        "location": model_dir
    }

    os.makedirs("metadata", exist_ok=True)
    with open(f"metadata/model_{metadata['version']}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata logged to metadata/model_{metadata['version']}.json")
