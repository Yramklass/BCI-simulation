import tf2onnx
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("best_model.h5")

# Manually set output_names for the Sequential model
if isinstance(model, tf.keras.Sequential):
    model.output_names = [f"output_{i}" for i in range(len(model.outputs))]

# Create an input signature from the model input shape
input_signature = [tf.TensorSpec([None] + list(model.input_shape[1:]), dtype=tf.float32, name="input")]

# Convert and save as ONNX
onnx_model_path = "eeg_cnn_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Model converted and saved as {onnx_model_path}")