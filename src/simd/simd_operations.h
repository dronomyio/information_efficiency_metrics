#pragma once

#include <vector>

class AutocorrelationSIMD {
public:
    std::vector<double> computeAutocorrelation(const double* returns, size_t n, size_t max_lag);
    std::pair<double, double> computeDecayParameters(const std::vector<double>& acf);
};

class BatchProcessor {
public:
    BatchProcessor(int threads = 0);
    std::vector<std::vector<double>> processBatch(
        const std::vector<std::vector<double>>& batch_returns,
        size_t max_lag
    );
};
