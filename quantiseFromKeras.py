import tensorflow as tf

# Load your trained .keras model
model = tf.keras.models.load_model("mobilenetv2_mendeley_26signs_augmented.keras")

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert to TFLite
tflite_model = converter.convert()

# Save quantized model
with open("mobilenetv2_mendeley_26signs_augmented_quant.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized model saved: mobilenetv2_mendeley_26signs_augmented_quant.tflite")
