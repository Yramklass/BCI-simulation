import tf2onnx
import tensorflow as tf

# Load saved model
model = tf.keras.models.load_model("eeg_cnn_model.h5")

# Convert and save as ONNX
onnx_model_path = "eeg_cnn_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model)
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Model converted and saved as {onnx_model_path}")
