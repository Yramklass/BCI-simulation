use ort::{Environment, Session, SessionBuilder, Value}; 
use ndarray::{Array3, CowArray}; 
use std::ffi::{CString, c_char};
use std::sync::Arc; 
use lazy_static::lazy_static;

use std::path::Path;
use std::env;

// Global Environment and ONNX Session
lazy_static! {
    // Create a global ONNX Runtime environment
    static ref ENVIRONMENT: Arc<Environment> = Arc::new(
        Environment::builder()
            .with_name("EEG_Classifier_Env")
            .build()
            .expect("Failed to create ONNX Runtime environment")
    );

    static ref SESSION: Session = {
        // Get the path to the ONNX model relative to Cargo.toml
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("best_attention_model.onnx")
            .to_string_lossy()
            .into_owned();

        println!("Loading ONNX model from: {}", model_path);  // Debug output

        SessionBuilder::new(&ENVIRONMENT)
            .expect("Failed to create SessionBuilder")
            .with_model_from_file(&model_path)
            .unwrap_or_else(|_| panic!("Failed to load ONNX model from {}", model_path))
    };
}

#[no_mangle]
pub extern "C" fn classify_eeg(input_ptr: *const f32) -> *mut c_char {
    // Define expected dimensions
    const TIMESTEPS: usize = 128;
    const FEATURES: usize = 44;
    const INPUT_LEN: usize = TIMESTEPS * FEATURES;

    // Convert input to Rust slice safely
    let input_slice = unsafe {
        if input_ptr.is_null() {
            eprintln!("Error: Received null pointer for EEG data.");
            return CString::new("Error: Null input").unwrap().into_raw(); 
        }
        std::slice::from_raw_parts(input_ptr, INPUT_LEN)
    };

    // Create 3D array [batch=1, timesteps=128, features=44]
    let input_array = match Array3::from_shape_vec((1, TIMESTEPS, FEATURES), input_slice.to_vec()) {
        Ok(arr) => arr,
        Err(e) => {
            eprintln!("Error creating ndarray: {}", e);
            return CString::new(format!("Error: Array shape mismatch - {}", e)).unwrap().into_raw();
        }
    };

    // Create ONNX tensor using CowArray AND .into_dyn()
    let cow_input_array = CowArray::from(&input_array);
    // Convert to dynamic dimensions as required by Value::from_array
    let dynamic_input_array = cow_input_array.into_dyn();

    let input_tensor = match Value::from_array(SESSION.allocator(), &dynamic_input_array) {
         Ok(val) => val,
         Err(e) => {
            eprintln!("Error creating ONNX tensor: {}", e);
            return CString::new(format!("Error: ONNX tensor creation failed - {}", e)).unwrap().into_raw();
         }
    };

    // Run inference, handle potential errors
    let outputs = match SESSION.run(vec![input_tensor]) {
        Ok(outputs) => outputs,
        Err(e) => {
            eprintln!("Error during ONNX inference: {}", e);
            return CString::new(format!("Error: Inference failed - {}", e)).unwrap().into_raw();
        }
    };

    // Extract the output tensor, handle potential errors
    let output_value = &outputs[0];
    let output_tensor = match output_value.try_extract::<f32>() {
         Ok(tensor) => tensor,
         Err(e) => {
            eprintln!("Error extracting output tensor: {}", e);
            return CString::new(format!("Error: Output extraction failed - {}", e)).unwrap().into_raw();
         }
    };

    // Get the prediction value
    let output_view = output_tensor.view();
    let prediction = match output_view.iter().next() {
        Some(val) => *val,
        None => {
            eprintln!("Error: Output tensor is empty.");
            return CString::new("Error: Empty output tensor").unwrap().into_raw();
        }
    };

    // Determine label based on the prediction
    let label = if prediction > 0.5 { "right" } else { "left" };

    // Return prediction label as a C string
    match CString::new(label) {
        Ok(c_string) => c_string.into_raw(),
        Err(e) => {
             eprintln!("Error creating CString: {}", e);
             CString::new("Error: Failed to create CString").unwrap().into_raw()
        }
    }
}

/// # Safety
/// This function frees a C string that was previously allocated by Rust
/// and passed to C. The pointer must not be null and must point to memory
/// allocated via `CString::into_raw`. Calling this with any other pointer
/// will lead to undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    let _ = CString::from_raw(ptr);
}