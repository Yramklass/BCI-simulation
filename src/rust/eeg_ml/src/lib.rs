use ort::{Session, Environment};
use ndarray::Array2;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// Load ONNX model globally
static MODEL_PATH: &str = "eeg_cnn_model.onnx";
lazy_static::lazy_static! {
    static ref SESSION: Session = Session::builder()
        .unwrap()
        .with_model_from_file(MODEL_PATH)
        .unwrap();
}

// Function to classify EEG data (called from C++)
#[no_mangle]
pub extern "C" fn classify_eeg(input_ptr: *const f32, len: usize) -> *mut c_char {
    let input_slice = unsafe { std::slice::from_raw_parts(input_ptr, len) };
    let input_array = Array2::from_shape_vec((1, len), input_slice.to_vec()).unwrap();

    let outputs = SESSION.run(vec![input_array.into_dyn()]).unwrap();
    let prediction: f32 = *outputs[0].as_slice().unwrap();

    let label = if prediction > 0.5 { "right" } else { "left" };
    CString::new(label).unwrap().into_raw()
}
