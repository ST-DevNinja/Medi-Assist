# train_models.py

from predictor import train_model


# Train models for all 3 types
train_model("lung")
train_model("prostate")
train_model("skin")

print("All models trained and saved successfully.")
