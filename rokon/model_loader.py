import tensorflow as tf
import inspect
from rokon import RNet  # Import RNet module properly

# Dynamically discover all functions in RNet.py
def get_model_dict():
    model_dict = {}
    # Loop through all members of the RNet module
    for name, obj in inspect.getmembers(RNet):
        if inspect.isfunction(obj):  # Check if the member is a function
            print(f"Found model: {name}")  # Debug print to check the model names
            model_dict[name] = obj
    return model_dict

# Get the dynamic model dictionary
model_dict = get_model_dict()

def get_rokon_model(model_name, input_shape=(256, 256, 3), num_classes=10, activation="relu"):
    if model_name in model_dict:
        return model_dict[model_name](input_shape=input_shape, num_classes=num_classes, activation=activation)
    else:
        raise ValueError(f"Model '{model_name}' not found in rokon! Available models: {', '.join(model_dict.keys())}")
