from src.data.loaders import load_data
from src.data.preprocess import preprocess_data
from src.data.split import split_data
from src.binning.woe_binning import perform_woe_binning
from src.features.selection import select_features
from src.features.generation import generate_features
from src.modeling.train import train_model
from src.evaluation.metrics import calculate_metrics
from src.evaluation.report import generate_report

def data_pipeline(file_path):
    # Load data
    data = load_data(file_path)
    
    # Preprocess data
    cleaned_data = preprocess_data(data)
    
    # Split data into training and testing sets
    train_data, test_data = split_data(cleaned_data)
    
    return train_data, test_data

def feature_pipeline(train_data):
    # Perform WOE binning
    binned_data = perform_woe_binning(train_data)
    
    # Generate new features
    features = generate_features(binned_data)
    
    # Select important features
    selected_features = select_features(features)
    
    return selected_features

def modeling_pipeline(train_data, selected_features):
    # Train the model
    model = train_model(train_data[selected_features])
    
    return model

def evaluation_pipeline(model, test_data, selected_features):
    # Calculate metrics
    metrics = calculate_metrics(model, test_data[selected_features])
    
    # Generate evaluation report
    report = generate_report(metrics)
    
    return report

def run_pipeline(file_path):
    train_data, test_data = data_pipeline(file_path)
    selected_features = feature_pipeline(train_data)
    model = modeling_pipeline(train_data, selected_features)
    report = evaluation_pipeline(model, test_data, selected_features)
    
    return report