#pragma once

#include <string>
#include <vector>

class PolygonDataReader {
public:
    PolygonDataReader(const std::string& api_key);
    ~PolygonDataReader();
    
    std::vector<double> computeReturns(const std::vector<double>& prices, int64_t interval_ns);
};
