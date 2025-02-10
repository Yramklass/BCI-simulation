#include <iostream>
#include "ffi_bridge.h"

int main() {
    float test_signal = 0.6;
    int prediction = call_rust_model(test_signal);

    if (prediction == 1)
        std::cout << "🔵 Move Right\n";
    else
        std::cout << "🔴 Move Left\n";

    return 0;
}
