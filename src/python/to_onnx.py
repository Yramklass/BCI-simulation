import tensorflow as tf
import tf2onnx
import numpy as np

# ---- STEP 1: Register Custom Loss ----
@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.math.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

# Update custom objects
tf.keras.utils.get_custom_objects().update({'focal_loss': focal_loss})

# ---- STEP 2: Load Model with Correct Input Shape ----
try:
    # Try loading normally first
    model = tf.keras.models.load_model("best_attention_model.keras")
except ValueError as e:
    print(f"Loading failed: {e}")
    print("Attempting to rebuild model architecture...")
    
    # Rebuild model with correct architecture
    def build_model():
        inputs = tf.keras.Input(shape=(128, 44))  # YOUR ACTUAL INPUT SHAPE
        
        # Reconstruct your exact architecture here
        x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='swish')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        
        # Continue with all other layers...
        # ... include all layers from your original model
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        return tf.keras.Model(inputs, outputs)
    
    model = build_model()
    model.load_weights("best_attention_model.keras")  # Load only weights

# ---- STEP 3: Verify Input Shape ----
print(f"Model input shape: {model.input_shape}")  # Should show (None, 128, 44)

# ---- STEP 4: Convert to ONNX ----
input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path="best_attention_model.onnx"
)

print(f"ONNX conversion successful. Saved to best_attention_model.onnx")