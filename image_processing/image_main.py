# hub.login('bf1c4516cac86a781b95844988e3da100be6c98ae5')

from ultralytics import YOLO

model = YOLO("../Models/model_1.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.predict(source="0",show = True)

print(results)













