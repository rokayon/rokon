import tensorflow as tf
import inspect
from rokon import RNet  # Explicitly import the RNet module

# Dynamically discover all classes in RNet.py
def get_model_dict():
    model_dict = {}
    # Loop through all members of the current module (RNet.py)
    for name, obj in inspect.getmembers(RNet):
        if inspect.isclass(obj):  # Check if the member is a class
            model_dict[name] = obj
    return model_dict

# Get the dynamic model dictionary
model_dict = get_model_dict()

def get_rokon_model(model_name, input_shape=(256, 256, 3), num_classes=10):
    if model_name in model_dict:
        return model_dict[model_name](input_shape=input_shape, num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not found in rokon!")
