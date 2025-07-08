# train_models.py

from predictor import train_model



# Train models for all 3 types
train_model("lung")
train_model("prostate")
train_model("skin")
train_model("heart")
train_model("kidney")
train_model("liver")
train_model("asthma")


print("All models trained and saved successfully.")
