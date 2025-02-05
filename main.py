from rokon.model_loader import get_rokon_model 
 
model_name = "RNet1" 
model = get_rokon_model(model_name, input_shape=(128, 128, 3), num_classes=5) 
 
model.summary() 
