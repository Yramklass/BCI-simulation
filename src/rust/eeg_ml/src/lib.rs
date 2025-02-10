#[no_mangle]
pub extern "C" fn classify_signal(value: f32) -> i32 {
    if value > 0.5 {
        1  // Move Right
    } else {
        -1 // Move Left
    }
}
