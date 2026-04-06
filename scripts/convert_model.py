from tensorflow.keras.models import load_model

model = load_model("model/malaria_model.h5", compile=False)

model.save("model/malaria_model.keras")

print("Model converted successfully!")
