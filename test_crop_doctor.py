from modules.crop_doctor import predict_image

image_path = "test_wheat.jpg"

result = predict_image(image_path)

print("Prediction:", result["display_name"])
print("Confidence:", result["confidence"], "%")
print("Probabilities:", result["probabilities"])
print("Device:", result["device"])