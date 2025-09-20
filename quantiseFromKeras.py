import tensorflow as tf

# Load your trained .keras model
model = tf.keras.models.load_model("models/mobilenetv2_mendeley_24signs_augmented.keras")

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert to TFLite
tflite_model = converter.convert()

# Save quantized model
with open("models/mobilenetv2_mendeley_24signs_augmented.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized model saved")
