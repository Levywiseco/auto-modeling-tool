from src.data.loaders import load_data
from src.data.preprocess import preprocess_data
from src.data.split import split_data
from src.binning.woe_binning import apply_woe_binning
from src.features.selection import select_features
from src.features.generation import generate_features
from src.modeling.train import train_model
from src.evaluation.metrics import evaluate_model
from src.evaluation.report import generate_report

def auto_pipeline(config):
    # Load data
    data = load_data(config['data_path'])
    
    # Preprocess data
    processed_data = preprocess_data(data)
    
    # Split data into training and testing sets
    train_data, test_data = split_data(processed_data, config['test_size'])
    
    # Apply binning techniques
    binned_data = apply_woe_binning(train_data, config['binning_params'])
    
    # Generate features
    features = generate_features(binned_data)
    
    # Select important features
    selected_features = select_features(features, config['feature_selection_params'])
    
    # Train the model
    model = train_model(selected_features, config['model_params'])
    
    # Evaluate the model
    evaluation_metrics = evaluate_model(model, test_data)
    
    # Generate evaluation report
    generate_report(evaluation_metrics, config['report_path'])

if __name__ == "__main__":
    import yaml
    
    with open('configs/default.yaml') as config_file:
        config = yaml.safe_load(config_file)
    
    auto_pipeline(config)