#include <vector>
#include <tuple>
#include "model.cpp"

class MPCController {
    std::tuple<double, double, double> computeControl(const VehicleState& state, const std::vector<PathPoint>& trajectory, double dt);
};
