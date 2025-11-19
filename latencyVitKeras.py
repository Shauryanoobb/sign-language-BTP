import numpy as np
import tensorflow as tf
from keras_vision.MobileViT_v1 import build_MobileViT_v1
from tensorflow.keras import layers, Model
import time
from statistics import mean, stdev, median
import string

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 256
NUM_CLASSES = 26
MODEL_PATH = "MobileVit-XXS-ASL-Augmented-TRUE-Mendeley.keras"
CLASS_NAMES = list(string.ascii_uppercase)

# Benchmark settings
WARMUP_RUNS = 20
BENCHMARK_RUNS = 200

# -----------------------------
# REBUILD MODEL ARCHITECTURE
# -----------------------------
print("🔧 Building model architecture...")

# Build MobileViT backbone
backbone = build_MobileViT_v1(
    model_type="XXS",
    pretrained=False,
    include_top=False,
    num_classes=0
)

# Define normalization and augmentation
normalization_layer = layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomRotation(0.05),
])

# Build the full model
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = normalization_layer(inputs)
x = data_augmentation(x)
x = backbone(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, outputs=x, name="MobileViT_ASL")

# Initialize model
print("🔨 Initializing model...")
dummy_input = tf.random.uniform((1, IMG_SIZE, IMG_SIZE, 3))
model(dummy_input)

# Load weights
print(f"📦 Loading weights from {MODEL_PATH}...")
try:
    model.load_weights(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    print("⚠️  Continuing with random weights for latency testing...")

model.trainable = False

# -----------------------------
# MODEL INFO
# -----------------------------
print(f"\n📊 Model Summary:")
print(f"   Model name: {model.name}")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Total params: {model.count_params():,}")
print(f"   Classes: {NUM_CLASSES} ({CLASS_NAMES[0]}-{CLASS_NAMES[-1]})")

# -----------------------------
# CREATE DUMMY INPUT
# -----------------------------
print(f"\n🎯 Creating dummy input with shape: (1, {IMG_SIZE}, {IMG_SIZE}, 3)")
# Simulating RGB image data (0-255 range, as webcam provides)
dummy_input = np.random.randint(0, 256, (1, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8).astype('float32')

# -----------------------------
# WARMUP PHASE
# -----------------------------
print(f"\n🔥 Warming up with {WARMUP_RUNS} iterations...")
for i in range(WARMUP_RUNS):
    _ = model.predict(dummy_input, verbose=0)
    if (i + 1) % 5 == 0:
        print(f"   Warmup progress: {i + 1}/{WARMUP_RUNS}")

# -----------------------------
# BENCHMARK PHASE
# -----------------------------
print(f"\n⏱️  Running {BENCHMARK_RUNS} benchmark iterations...")
latencies_ms = []

for i in range(BENCHMARK_RUNS):
    start = time.perf_counter()
    predictions = model.predict(dummy_input, verbose=0)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    latencies_ms.append(latency_ms)
    
    if (i + 1) % 20 == 0:
        current_avg = mean(latencies_ms)
        print(f"   Progress: {i + 1}/{BENCHMARK_RUNS} | Current: {latency_ms:.2f} ms | Avg so far: {current_avg:.2f} ms")

# -----------------------------
# CALCULATE STATISTICS
# -----------------------------
avg_latency = mean(latencies_ms)
std_latency = stdev(latencies_ms) if len(latencies_ms) > 1 else 0
median_latency = median(latencies_ms)
min_latency = min(latencies_ms)
max_latency = max(latencies_ms)
fps = 1000 / avg_latency

# Calculate percentiles
latencies_sorted = sorted(latencies_ms)
p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
print("\n" + "=" * 70)
print("📈 LATENCY BENCHMARK RESULTS")
print("=" * 70)
print(f"Model:                  {model.name}")
print(f"Task:                   ASL Sign Language Recognition")
print(f"Number of runs:         {BENCHMARK_RUNS}")
print(f"Input shape:            {dummy_input.shape}")
print("-" * 70)
print(f"Average latency:        {avg_latency:.2f} ms")
print(f"Median latency:         {median_latency:.2f} ms")
print(f"Standard deviation:     {std_latency:.2f} ms")
print("-" * 70)
print(f"Min latency:            {min_latency:.2f} ms")
print(f"Max latency:            {max_latency:.2f} ms")
print("-" * 70)
print(f"50th percentile (P50):  {p50:.2f} ms")
print(f"95th percentile (P95):  {p95:.2f} ms")
print(f"99th percentile (P99):  {p99:.2f} ms")
print("-" * 70)
print(f"Throughput (FPS):       {fps:.2f} FPS")
print(f"Real-time @ 30 FPS:     {'✅ Yes' if avg_latency < 33.33 else '❌ No'} (< 33.33 ms needed)")
print(f"Real-time @ 15 FPS:     {'✅ Yes' if avg_latency < 66.67 else '❌ No'} (< 66.67 ms needed)")
print("=" * 70)

# -----------------------------
# LATENCY DISTRIBUTION
# -----------------------------
print("\n📊 LATENCY DISTRIBUTION:")
bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, float('inf')]
bin_labels = ['0-10ms', '10-20ms', '20-30ms', '30-40ms', '40-50ms', '50-75ms', '75-100ms', '100-150ms', '150ms+']
counts = [0] * len(bin_labels)

for lat in latencies_ms:
    for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
        if lower <= lat < upper:
            counts[i] += 1
            break

for label, count in zip(bin_labels, counts):
    percentage = (count / BENCHMARK_RUNS) * 100
    bar = '█' * int(percentage / 2)
    print(f"{label:12s}: {bar:50s} {count:3d} ({percentage:5.1f}%)")

# -----------------------------
# WEBCAM PERFORMANCE ANALYSIS
# -----------------------------
print("\n📹 WEBCAM REAL-TIME PERFORMANCE:")
print("-" * 70)

# Typical webcam FPS options
webcam_fps_options = [15, 24, 30, 60]
print(f"{'Webcam FPS':<15} {'Latency Budget':<18} {'Actual Latency':<18} {'Status'}")
print("-" * 70)

for fps_target in webcam_fps_options:
    latency_budget = 1000 / fps_target
    status = "✅ OK" if avg_latency < latency_budget else "❌ Too slow"
    print(f"{fps_target:<15} {latency_budget:<18.1f} ms {avg_latency:<18.2f} ms {status}")

print("-" * 70)

# -----------------------------
# USER EXPERIENCE ANALYSIS
# -----------------------------
print("\n👤 USER EXPERIENCE ANALYSIS:")
print("-" * 70)

if avg_latency < 16.67:
    ux_rating = "🌟 EXCELLENT"
    ux_desc = "Feels instant, perfect for real-time finger spelling"
elif avg_latency < 33.33:
    ux_rating = "✅ GOOD"
    ux_desc = "Smooth experience, suitable for continuous recognition"
elif avg_latency < 66.67:
    ux_rating = "⚠️  ACCEPTABLE"
    ux_desc = "Noticeable delay, but usable for practice"
elif avg_latency < 100:
    ux_rating = "⚠️  POOR"
    ux_desc = "Significant lag, may frustrate users"
else:
    ux_rating = "❌ UNUSABLE"
    ux_desc = "Too slow for interactive applications"

print(f"UX Rating:  {ux_rating}")
print(f"Experience: {ux_desc}")
print(f"Response:   User will perceive ~{avg_latency:.0f}ms delay between sign and prediction")
print("-" * 70)

# -----------------------------
# OPTIMIZATION OPPORTUNITIES
# -----------------------------
print("\n🔧 OPTIMIZATION OPPORTUNITIES:")
print("-" * 70)

# Calculate preprocessing overhead (rough estimate)
preprocessing_time = 2.0  # Typical time for resize + color conversion
inference_time = avg_latency - preprocessing_time

print(f"Estimated breakdown:")
print(f"  • Preprocessing (resize/color): ~2-3 ms")
print(f"  • Model inference:              ~{inference_time:.1f} ms")
print(f"  • Total:                        {avg_latency:.2f} ms")
print()

if avg_latency > 33.33:
    print("💡 Suggestions to improve latency:")
    print("   1. Convert to TensorFlow Lite (.tflite)")
    print("   2. Apply post-training quantization (INT8)")
    print("   3. Reduce input size from 256x256 to 224x224 or 192x192")
    print("   4. Use GPU acceleration if available")
    print("   5. Remove data augmentation layers during inference")
    print("   6. Consider pruning/distillation techniques")
else:
    print("✅ Current latency is already excellent for real-time use!")
    print("   Optional optimizations:")
    print("   • TFLite conversion for mobile deployment")
    print("   • Quantization to reduce model size")

print("-" * 70)

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
print("\n💡 DEPLOYMENT RECOMMENDATIONS:")
print("-" * 70)

if avg_latency < 16.67:
    print("✅ Perfect for:")
    print("   • Real-time finger spelling applications")
    print("   • Interactive ASL learning tools")
    print("   • High-FPS webcam capture (60 FPS)")
    print("   • Mobile deployment with minor optimizations")
elif avg_latency < 33.33:
    print("✅ Suitable for:")
    print("   • Standard webcam applications (30 FPS)")
    print("   • Desktop ASL recognition tools")
    print("   • Educational software")
    print("   ⚠️  May need TFLite for mobile deployment")
elif avg_latency < 66.67:
    print("⚠️  Limited use cases:")
    print("   • Low-FPS applications (15 FPS)")
    print("   • Non-interactive batch processing")
    print("   • Requires optimization for good UX")
    print("   ❌ Not recommended for mobile")
else:
    print("❌ Not suitable for real-time applications")
    print("   • Requires significant optimization")
    print("   • Consider model compression/pruning")
    print("   • Try smaller input size or backbone")

print("-" * 70)

# -----------------------------
# ADDITIONAL INSIGHTS
# -----------------------------
print("\n📊 ADDITIONAL INSIGHTS:")
print(f"   • Signs recognizable per minute: {60 * fps:.0f}")
print(f"   • Consistency (CV): {(std_latency/avg_latency)*100:.1f}% {'✅ Stable' if (std_latency/avg_latency) < 0.15 else '⚠️  Variable'}")
print(f"   • 95% of predictions complete within: {p95:.2f} ms")

print("\n✅ Benchmark completed!")
print("\n🔍 Next steps:")
print("   1. Test on actual webcam to measure end-to-end latency")
print("   2. Profile preprocessing overhead separately")
print("   3. Consider TFLite conversion if deploying to mobile/edge devices")