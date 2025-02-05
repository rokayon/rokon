import tensorflow as tf 
from tensorflow.keras import layers, models 
 
def RNet1(input_shape=(256, 256, 3), num_classes=10): 
    model = models.Sequential([ 
        layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape), 
        layers.MaxPooling2D(2,2), 
        layers.Conv2D(32, (3,3), activation='relu'), 
        layers.MaxPooling2D(2,2), 
        layers.Flatten(), 
        layers.Dense(128, activation='relu'), 
        layers.Dense(num_classes, activation='softmax') 
    ]) 
    return model 
 
def RNet2(input_shape=(256, 256, 3), num_classes=10): 
    model = models.Sequential([ 
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape), 
        layers.Conv2D(32, (3,3), activation='relu'), 
        layers.MaxPooling2D(2,2), 
        layers.Conv2D(64, (3,3), activation='relu'), 
        layers.MaxPooling2D(2,2), 
        layers.Flatten(), 
        layers.Dense(256, activation='relu'), 
        layers.Dense(num_classes, activation='softmax') 
    ]) 
    return model 
