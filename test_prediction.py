
from ml.predict import predict_image

# Replace this with the path to one wheat leaf image
IMAGE_PATH = "data/crop_disease/wheat/WheatLeafRust/WheatLeafRust/test/diseased/1.jpg"

result = predict_image(IMAGE_PATH)

print("\n========== CROP DOCTOR RESULT ==========")

print("Prediction:", result["display_name"])
print("Confidence:", result["confidence"], "%")

print("\nAll class probabilities:")

for disease, probability in result["probabilities"].items():
    print(f"{disease}: {probability}%")

print("========================================")
