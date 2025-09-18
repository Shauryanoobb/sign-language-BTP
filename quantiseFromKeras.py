import tensorflow as tf

# Load your trained .keras model
model = tf.keras.models.load_model("mobilenetv2_signlang_synthetic_14_signs.keras")

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert to TFLite
tflite_model = converter.convert()

# Save quantized model
with open("mobilenetv2_signlang_synthetic_14_signs_quant.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized model saved: mobilenetv2_signlang_synthetic_14_signs_quant.tflite")
