import tensorflow as tf
from keras_vision.MobileViT_v1 import build_MobileViT_v1
from tensorflow.keras import layers, Model

backbone = build_MobileViT_v1(model_type="XXS", pretrained=False, include_top=False, num_classes=0)

x = backbone.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(26, activation='softmax')(x)  # 26 ASL classes

model = Model(inputs=backbone.input, outputs=x)

# 2️⃣ Build the model by passing a dummy input
dummy_input = tf.random.uniform((1, 256, 256, 3))
model(dummy_input)  # ensures all layers have weights

# 3️⃣ Load weights from your trained file
MODEL_PATH = "MobileVit-XXS-ASL-Augmented-Mendeley.keras"
model.load_weights(MODEL_PATH)

# model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False
num_classes = model.output_shape[-1]

print("✅ Model loaded. Num classes:", num_classes)

# 1️⃣ Convert your trained MobileViT model to a TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Option A: Dynamic range quantization (simple)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("mobilevit_asl_dynamic.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Dynamic Range Quantized model saved!")
