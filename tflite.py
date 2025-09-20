#to get the .tflite model
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("mobilenetv2_signlang_synthetic_14_signs.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save to file
with open("mobilenetv2_signlang_synthetic_14_signs.tflite", "wb") as f:
    f.write(tflite_model)

print("tflite model saved")
