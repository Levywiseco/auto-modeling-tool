# This script is used to export trained models for deployment or further use.

import joblib
import os
import sys

def export_model(model, model_name, export_path):
    """
    Exports the trained model to the specified path using joblib.
    
    Parameters:
    - model: The trained model to be exported.
    - model_name: The name of the model file (without extension).
    - export_path: The directory where the model will be saved.
    """
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    
    model_file = os.path.join(export_path, f"{model_name}.pkl")
    joblib.dump(model, model_file)
    print(f"Model exported successfully to {model_file}")

if __name__ == "__main__":
    # Example usage: python export_model.py <model_name> <export_path>
    if len(sys.argv) != 3:
        print("Usage: python export_model.py <model_name> <export_path>")
        sys.exit(1)

    model_name = sys.argv[1]
    export_path = sys.argv[2]

    # Placeholder for the model; in practice, this would be loaded or passed in.
    model = None  # Replace with actual model instance

    export_model(model, model_name, export_path)