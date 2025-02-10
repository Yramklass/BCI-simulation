#include <iostream>
#include <windows.h>

typedef int (*ClassifySignalFn)(float);

int call_rust_model(float value) {
    // Convert narrow string to wide string
    const wchar_t* libPath = L"../rust/eeg_ml/target/release/eeg_ml.dll";

    HMODULE handle = LoadLibraryW(libPath);  // Use LoadLibraryW for wide string
    if (!handle) {
        std::cerr << "Cannot load Rust library. Error code: " << GetLastError() << std::endl;
        return 0;
    }

    ClassifySignalFn classify_signal = (ClassifySignalFn)GetProcAddress(handle, "classify_signal");
    if (!classify_signal) {
        std::cerr << "Cannot find function in Rust library. Error code: " << GetLastError() << std::endl;
        FreeLibrary(handle);
        return 0;
    }

    int result = classify_signal(value);
    FreeLibrary(handle);
    return result;
}
