import time
import numpy as np
import cv2
import tensorflow as tf

# ------------------------------
# CONFIG
# ------------------------------
IMG_SIZE = (112, 112)  # H, W
MODEL_PATH = "models/mobilevit_asl_dynamic.tflite"
NUM_WARMUP = 10  # Warmup runs to stabilize timing
NUM_ITERATIONS = 200  # Number of test iterations

CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y','Z',
]

# ------------------------------
# LOAD TFLITE MODEL
# ------------------------------
print(f"📦 Loading model: {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Model loaded successfully")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# ------------------------------
# TIMING STORAGE
# ------------------------------
inference_times = []

# ------------------------------
# PREPROCESS FUNCTION
# ------------------------------
def preprocess_random_image():
    """Generate random image and preprocess for TFLite."""
    # Generate random RGB image
    img = np.random.randint(0, 255, (IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Shape: (1, H, W, C)
    return img

# ------------------------------
# INFERENCE FUNCTION
# ------------------------------
def run_inference(img):
    """Run inference and measure time."""
    # ---- TIME START ----
    t1 = time.time()
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    # ---- TIME END ----
    t2 = time.time()
    
    # Timing in ms
    infer_ms = (t2 - t1) * 1000
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(output_data, axis=1)[0]
    confidence = float(output_data[0][pred_class])
    
    return CLASS_NAMES[pred_class], confidence, infer_ms

# ------------------------------
# WARMUP PHASE
# ------------------------------
print(f"\n🔥 Running {NUM_WARMUP} warmup iterations...")
for i in range(NUM_WARMUP):
    img = preprocess_random_image()
    run_inference(img)
print("✅ Warmup complete")

# ------------------------------
# BENCHMARK PHASE
# ------------------------------
print(f"\n🔬 Starting inference timing test ({NUM_ITERATIONS} iterations)...")
print("-" * 60)

for i in range(NUM_ITERATIONS):
    # Generate random input
    img = preprocess_random_image()
    
    # Run inference
    pred_class, confidence, infer_ms = run_inference(img)
    inference_times.append(infer_ms)
    
    # Print every 20 iterations
    if (i + 1) % 20 == 0:
        print(f"[{i+1:3d}/{NUM_ITERATIONS}] {pred_class} ({confidence:.2f})  |  Time = {infer_ms:.2f} ms")

# ------------------------------
# STATISTICS
# ------------------------------
print("\n" + "=" * 60)
print("INFERENCE LATENCY STATISTICS")
print("=" * 60)

if inference_times:
    inference_times = np.array(inference_times)
    
    print(f"Total inferences:     {len(inference_times)}")
    print(f"Mean inference time:  {np.mean(inference_times):.2f} ms")
    print(f"Median inference:     {np.median(inference_times):.2f} ms")
    print(f"Std deviation:        {np.std(inference_times):.2f} ms")
    print(f"Min inference time:   {np.min(inference_times):.2f} ms")
    print(f"Max inference time:   {np.max(inference_times):.2f} ms")
    print(f"\nPercentiles:")
    print(f"  P50 (median):       {np.percentile(inference_times, 50):.2f} ms")
    print(f"  P90:                {np.percentile(inference_times, 90):.2f} ms")
    print(f"  P95:                {np.percentile(inference_times, 95):.2f} ms")
    print(f"  P99:                {np.percentile(inference_times, 99):.2f} ms")
    
    # Throughput
    avg_time_s = np.mean(inference_times) / 1000
    throughput = 1.0 / avg_time_s if avg_time_s > 0 else 0
    print(f"\nThroughput:           {throughput:.2f} inferences/sec")
else:
    print("❌ No inferences were made.")

print("\n✅ Benchmark complete!")