#to get the .tflite model
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("tinyvit_student_final_synthetic.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save to file
with open("asl_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved asl_model.tflite")
