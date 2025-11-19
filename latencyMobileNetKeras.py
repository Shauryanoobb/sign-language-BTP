import tensorflow as tf
import numpy as np
import time
import statistics

# -------------------------------
# Load saved model (.keras)
# -------------------------------
model_path = "models/MobileVit-XXS-ASL-Augmented-Mendeley.keras"   # change this
model = tf.keras.models.load_model(model_path)
print("Model loaded.")

# -------------------------------
# Dummy input (same as training)
# -------------------------------
dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32)

# Warm-up iterations (important for GPU)
for _ in range(10):
    _ = model(dummy_input)

# -------------------------------
# Measure latency
# -------------------------------
latencies = []
iterations = 200

print("Running latency test…")

for _ in range(iterations):
    start = time.time()
    _ = model(dummy_input)
    end = time.time()
    latencies.append((end - start) * 1000)  # ms

# -------------------------------
# Results
# -------------------------------
avg = sum(latencies) / len(latencies)
p50 = statistics.median(latencies)
p90 = np.percentile(latencies, 90)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)

print("\n===== Latency Results (ms) =====")
print(f"Iterations        : {iterations}")
print(f"Average latency   : {avg:.3f} ms")
print(f"Median  (p50)     : {p50:.3f} ms")
print(f"p90 latency       : {p90:.3f} ms")
print(f"p95 latency       : {p95:.3f} ms")
print(f"p99 latency       : {p99:.3f} ms")
print("=================================")
