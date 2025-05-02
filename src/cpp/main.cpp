#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

// Declare Rust functions
extern "C" {
    char* classify_eeg(const float* input);
    void free_string(char* ptr);
}

// Constants matching your model
const int WINDOW_SIZE = 128;
const int NUM_FEATURES = 44; // 22 channels × 2 (raw + delta)

// Function to load CSV data
std::vector<std::vector<float>> loadCSV(const std::string& filename) {  
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }

        data.push_back(row);
    }

    return data;
}

int main() {
    // 1. Load your EEG data from CSV
    std::string csv_file = "eeg_data.csv";
    auto eeg_data = loadCSV(csv_file);

    // 2. Verify data dimensions
    if (eeg_data.size() < WINDOW_SIZE || eeg_data[0].size() != NUM_FEATURES) {
        std::cerr << "Error: Data must have at least " << WINDOW_SIZE 
                  << " rows and exactly " << NUM_FEATURES << " columns\n";
        return 1;
    }

    // 3. Prepare input buffer
    std::vector<float> model_input;
    model_input.reserve(WINDOW_SIZE * NUM_FEATURES);

    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            model_input.push_back(eeg_data[t][f]);
        }
    }

    // 4. Call Rust classifier
    char* prediction_ptr = classify_eeg(model_input.data());
    
    if (prediction_ptr != nullptr) {
        std::string prediction_str(prediction_ptr);
        std::cout << "Model prediction: " << prediction_str << std::endl;
        
        if (prediction_str.rfind("Error:", 0) == 0) {
            std::cerr << "Warning: Received error from classification library." << std::endl;
        }
        free_string(prediction_ptr);
    } else {
        std::cerr << "Error: Received null pointer from classification library." << std::endl;
    }

    return 0;
}