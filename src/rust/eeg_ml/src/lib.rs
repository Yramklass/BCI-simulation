use ort::{Session, Environment, Value};
use ndarray::{Array, IxDyn};
use std::ffi::{CString, c_char};
use std::os::raw::c_char;
use lazy_static::lazy_static;

// Load ONNX model globally
static MODEL_PATH: &str = "eeg_cnn_model.onnx";

lazy_static! {
    static ref SESSION: Session = Session::builder()
        .unwrap()
        .with_model_from_file(MODEL_PATH)
        .unwrap();
}

// Function to classify EEG data (called from C++)
#[no_mangle]
pub extern "C" fn classify_eeg(input_ptr: *const f32, len: usize) -> *mut c_char {
    // Convert raw pointer to Rust slice
    let input_slice = unsafe { std::slice::from_raw_parts(input_ptr, len) };
    
    // Convert to ndarray with dynamic shape (assuming 1 batch)
    let input_array = Array::from_shape_vec(IxDyn(&[1, len]), input_slice.to_vec()).unwrap();

    // Wrap input in ONNX tensor format
    let input_tensor = Value::from_array(SESSION.allocator(), &input_array).unwrap();

    // Run inference
    let outputs = SESSION.run(vec![input_tensor]).unwrap();

    // Extract prediction (assuming single output)
    let output_tensor = outputs[0].extract_tensor::<f32>().unwrap();
    let prediction = output_tensor.view().iter().cloned().next().unwrap();

    // Convert float prediction to label
    let label = if prediction > 0.5 { "right" } else { "left" };

    // Return label as C string
    CString::new(label).unwrap().into_raw()
}
