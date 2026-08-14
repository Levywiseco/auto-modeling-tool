from auto_modeling_tool.binning.woe_binning import perform_woe_binning
from auto_modeling_tool.data.loaders import load_data
from auto_modeling_tool.data.preprocess import preprocess_data
from auto_modeling_tool.data.split import split_data
from auto_modeling_tool.evaluation.metrics import calculate_metrics
from auto_modeling_tool.evaluation.report import generate_report
from auto_modeling_tool.features.generation import generate_features
from auto_modeling_tool.features.selection import select_features
from auto_modeling_tool.modeling.train import train_model


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
