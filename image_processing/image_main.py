from ultralytics import YOLO, checks, hub
checks()

hub.login('bf1c4516cac86a781b95844988e3da100be6c98ae5')

model = YOLO('https://hub.ultralytics.com/models/SHI2Pcl6fHBO3q8CKY2a')
results = model.train()