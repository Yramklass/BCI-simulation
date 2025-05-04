import tensorflow as tf
import tf2onnx
import numpy as np
import os

print(f"TensorFlow version: {tf.__version__}")
print(f"tf2onnx version: {tf2onnx.__version__}")


@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss function. Ensure alpha and gamma defaults match
    the parameters used during model compilation if not overridden.
    The implementation should be numerically stable.
    """
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # Calculate cross-entropy
    cross_entropy_true = -y_true * tf.math.log(y_pred)
    cross_entropy_false = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)

    # Calculate loss for true classes (pt)
    pt_true = y_pred
    loss_true = alpha * tf.math.pow(1.0 - pt_true, gamma) * cross_entropy_true

    # Calculate loss for false classes (1-pt)
    pt_false = 1.0 - y_pred
    loss_false = (1.0 - alpha) * tf.math.pow(1.0 - pt_false, gamma) * cross_entropy_false 
    ce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
    p_t = (y_true * y_pred) + ((1-y_true) * (1-y_pred))
    alpha_factor = y_true * alpha + (1-y_true) * (1-alpha)
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    loss = tf.reduce_mean(alpha_factor * modulating_factor * ce) # Mean over batch
    return loss

keras_model_path = "best_attention_model.keras"
onnx_model_path = "best_attention_model.onnx"
opset_version = 13

custom_objects = {'focal_loss': focal_loss}
tf.keras.utils.get_custom_objects().update(custom_objects)
print("Custom objects registered.")

print(f"Attempting to load Keras model from: {keras_model_path}")

if not os.path.exists(keras_model_path):
    print(f"ERROR: Keras model file not found at {keras_model_path}")
    exit()

try:
    # Load the model, ensuring custom objects are known
    model = tf.keras.models.load_model(
        keras_model_path,
        custom_objects=custom_objects,
        compile=False 
    )
    model.summary() # Print model summary to verify architecture
    print("Keras model loaded successfully.")
    print(f"Expected model input shape: {model.input_shape}")

except Exception as e:
    print(f"\nERROR: Failed to load Keras model.")
    print(f"Specific Error: {e}")
    print("\nTroubleshooting tips:")
    print("1. Verify the 'keras_model_path' is correct.")
    print("2. Ensure the 'focal_loss' function definition above EXACTLY matches the one used during training.")
    print("3. Check for TensorFlow version compatibility issues between training and conversion environments.")
    print("4. Make sure the .keras file is not corrupted.")
    exit() # Stop the script if loading fails

try:
    input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input")]
    print(f"Input signature for ONNX conversion: {input_signature}")
except Exception as e:
    print(f"ERROR: Could not determine input signature from model: {e}")
    exit()

# --- Convert to ONNX ---
print(f"\nAttempting to convert model to ONNX (Opset {opset_version})...")
try:
    onnx_model, _ = tf2onnx.convert.from_keras(
        model=model,
        input_signature=input_signature,
        opset=opset_version,
        output_path=onnx_model_path
    )
    print(f"ONNX conversion successful!")
    print(f"Model saved to: {onnx_model_path}")

except Exception as e:
    print(f"\nERROR: ONNX conversion failed.")
    print(f"Specific Error: {e}")
    print("\nTroubleshooting tips:")
    print("1. Try a different 'opset_version' (e.g., 11, 12, 14, 15). Some layers are better supported in different opsets.")
    print("2. Check if any layers in your model are known to be problematic for tf2onnx conversion.")
    print("3. Ensure input signature matches the model's expectation.")
    print("4. Update tf2onnx and tensorflow to their latest compatible versions.")