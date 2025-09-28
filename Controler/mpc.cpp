#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "model.cpp"

class MPCController {
private:
    // MPC parameters
    int prediction_horizon;     // N - number of prediction steps
    int control_horizon;        // Nc - number of control steps (usually <= N)
    double dt;                 // time step
    
    // Vehicle parameters
    double wheelbase;
    double max_steering_angle;
    double max_acceleration;
    double max_deceleration;
    double max_throttle;
    double max_brake;
    
    // Cost function weights
    struct CostWeights {
        double w_cte;          // Cross-track error weight
        double w_epsi;         // Heading error weight
        double w_v;            // Velocity error weight
        double w_delta;        // Steering angle penalty
        double w_a;            // Acceleration penalty
        double w_delta_d;      // Steering rate penalty
        double w_a_d;          // Acceleration rate penalty
    } weights;
    
    // Previous control inputs (for rate penalties)
    double prev_steering;
    double prev_acceleration;
    
public:
    MPCController(int N = 10,                    // Prediction horizon
                 int Nc = 10,                   // Control horizon 
                 double dt_val = 0.1,           // Time step
                 double wb = 2.7,               // Wheelbase
                 double max_steer = 0.5236,     // Max steering (30 deg)
                 double max_accel = 3.0,        // Max acceleration
                 double max_decel = 5.0,        // Max deceleration
                 double max_throttle_val = 1.0, // Max throttle
                 double max_brake_val = 1.0)    // Max brake
        : prediction_horizon(N), control_horizon(Nc), dt(dt_val),
          wheelbase(wb), max_steering_angle(max_steer),
          max_acceleration(max_accel), max_deceleration(max_decel),
          max_throttle(max_throttle_val), max_brake(max_brake_val),
          prev_steering(0.0), prev_acceleration(0.0) {
        
        // Initialize cost weights - these can be tuned
        weights.w_cte = 2000.0;      // High weight on cross-track error
        weights.w_epsi = 2000.0;     // High weight on heading error
        weights.w_v = 1.0;           // Weight on velocity error
        weights.w_delta = 5.0;       // Penalty on steering magnitude
        weights.w_a = 5.0;           // Penalty on acceleration magnitude
        weights.w_delta_d = 200.0;   // High penalty on steering rate
        weights.w_a_d = 10.0;        // Penalty on acceleration rate
    }
    
    //Kinematic bicycle model prediction
    VehicleState predictState(const VehicleState& state, double throttle, double brake, double steering_angle, double dt_step) {
        VehicleState next_state = state;
        
        // Clamp steering
        steering_angle = std::max(-max_steering_angle, 
                                 std::min(max_steering_angle, steering_angle));
        
        // Update position
        next_state.x += state.v * std::cos(state.theta) * dt_step;
        next_state.y += state.v * std::sin(state.theta) * dt_step;
        
        // Update heading
        next_state.theta += (state.v / wheelbase) * std::tan(steering_angle) * dt_step;
        
        // Normalize heading
        while (next_state.theta > M_PI) next_state.theta -= 2 * M_PI;
        while (next_state.theta < -M_PI) next_state.theta += 2 * M_PI;
        
        // Update velocity
        double acceleration = 0.0;
        if (throttle > 0) {
            acceleration = throttle * max_acceleration;
        } else if (brake > 0) {
            acceleration = -brake * max_deceleration;
        }
        
        next_state.v += acceleration * dt_step;
        next_state.v = std::max(0.0, next_state.v);
        
        return next_state;
    }
    
    std::pair<double, double> calculateErrors(const VehicleState& state, const std::vector<PathPoint>& path) {
        
        int closest_idx = 0;
        double min_distance = std::numeric_limits<double>::max();
        
        for (int i = 0; i < path.size(); ++i) {
            double dx = state.x - path[i].x;
            double dy = state.y - path[i].y;
            double distance = std::sqrt(dx*dx + dy*dy);
            
            if (distance < min_distance) {
                min_distance = distance;
                closest_idx = i;
            }
        }
        
        const PathPoint& closest_point = path[closest_idx];
        
        // cross-track error
        double dx = state.x - closest_point.x;
        double dy = state.y - closest_point.y;
        double cte = -std::sin(closest_point.theta) * dx + std::cos(closest_point.theta) * dy;
        
        // Heading error
        double epsi = state.theta - closest_point.theta;
        
        while (epsi > M_PI) epsi -= 2 * M_PI;
        while (epsi < -M_PI) epsi += 2 * M_PI;
        
        return {cte, epsi};
    }

    double evaluateCost(const VehicleState& initial_state, const std::vector<PathPoint>& reference_path, const std::vector<double>& steering_sequence, const std::vector<double>& throttle_sequence, const std::vector<double>& brake_sequence, double target_velocity) {

        double total_cost = 0.0;
        VehicleState current_state = initial_state;
        
        // Predict trajectory and accumulate costs
        for (int i = 0; i < prediction_horizon; ++i) {
            double steering = (i < steering_sequence.size()) ? steering_sequence[i] : 0.0;
            double throttle = (i < throttle_sequence.size()) ? throttle_sequence[i] : 0.0;
            double brake = (i < brake_sequence.size()) ? brake_sequence[i] : 0.0;
            
            // Calculate state errors
            auto [cte, epsi] = calculateErrors(current_state, reference_path);
            double velocity_error = current_state.v - target_velocity;
            
            total_cost += weights.w_cte * cte * cte;
            total_cost += weights.w_epsi * epsi * epsi;
            total_cost += weights.w_v * velocity_error * velocity_error;
    
            total_cost += weights.w_delta * steering * steering;
            
            double acceleration = 0.0;
            if (throttle > 0) acceleration = throttle * max_acceleration;
            else if (brake > 0) acceleration = -brake * max_deceleration;
            total_cost += weights.w_a * acceleration * acceleration;
            
            if (i > 0) {
                double prev_steering_cmd = (i-1 < steering_sequence.size()) ? 
                                          steering_sequence[i-1] : prev_steering;
                double prev_accel_cmd = 0.0;
                if (i-1 < throttle_sequence.size() && throttle_sequence[i-1] > 0) {
                    prev_accel_cmd = throttle_sequence[i-1] * max_acceleration;
                } else if (i-1 < brake_sequence.size() && brake_sequence[i-1] > 0) {
                    prev_accel_cmd = -brake_sequence[i-1] * max_deceleration;
                } else {
                    prev_accel_cmd = prev_acceleration;
                }
                
                total_cost += weights.w_delta_d * std::pow(steering - prev_steering_cmd, 2);
                total_cost += weights.w_a_d * std::pow(acceleration - prev_accel_cmd, 2);
            } else {
                total_cost += weights.w_delta_d * std::pow(steering - prev_steering, 2);
                total_cost += weights.w_a_d * std::pow(acceleration - prev_acceleration, 2);
            }

            current_state = predictState(current_state, throttle, brake, steering, dt);
        }
        
        return total_cost;
    }
    
    // Simple gradient-free optimization
    std::tuple<double, double, double> optimizeControl(const VehicleState& state,
                                                      const std::vector<PathPoint>& reference_path,
                                                      double target_velocity) {
        
        std::vector<double> best_steering(control_horizon, 0.0);
        std::vector<double> best_throttle(control_horizon, 0.0);
        std::vector<double> best_brake(control_horizon, 0.0);
        
        double best_cost = std::numeric_limits<double>::max();
        
        const int num_samples = 5;
        
        for (int steer_idx = 0; steer_idx < num_samples; ++steer_idx) {
            double steering = -max_steering_angle + 
                            (2.0 * max_steering_angle * steer_idx) / (num_samples - 1);
            
            for (int throttle_idx = 0; throttle_idx < num_samples; ++throttle_idx) {
                double throttle = (1.0 * throttle_idx) / (num_samples - 1);
 
                std::vector<double> steering_seq(control_horizon, steering);
                std::vector<double> throttle_seq(control_horizon, throttle);
                std::vector<double> brake_seq(control_horizon, 0.0);
                
                double cost = evaluateCost(state, reference_path, steering_seq, 
                                         throttle_seq, brake_seq, target_velocity);
                
                if (cost < best_cost) {
                    best_cost = cost;
                    best_steering = steering_seq;
                    best_throttle = throttle_seq;
                    best_brake = brake_seq;
                }
            }
            
            for (int brake_idx = 1; brake_idx < num_samples; ++brake_idx) {
                double brake = (1.0 * brake_idx) / (num_samples - 1);
                
                std::vector<double> steering_seq(control_horizon, steering);
                std::vector<double> throttle_seq(control_horizon, 0.0);
                std::vector<double> brake_seq(control_horizon, brake);
                
                double cost = evaluateCost(state, reference_path, steering_seq,
                                         throttle_seq, brake_seq, target_velocity);
                
                if (cost < best_cost) {
                    best_cost = cost;
                    best_steering = steering_seq;
                    best_throttle = throttle_seq;
                    best_brake = brake_seq;
                }
            }
        }
        
        if (!best_steering.empty()) prev_steering = best_steering[0];
        if (!best_throttle.empty() && !best_brake.empty()) {
            if (best_throttle[0] > 0) {
                prev_acceleration = best_throttle[0] * max_acceleration;
            } else if (best_brake[0] > 0) {
                prev_acceleration = -best_brake[0] * max_deceleration;
            }
        }
        
        double steering_cmd = best_steering.empty() ? 0.0 : best_steering[0];
        double throttle_cmd = best_throttle.empty() ? 0.0 : best_throttle[0];
        double brake_cmd = best_brake.empty() ? 0.0 : best_brake[0];
        
        return {steering_cmd, throttle_cmd, brake_cmd};
    }
    
    std::tuple<double, double, double> computeControl(const VehicleState& state, const std::vector<PathPoint>& trajectory, double target_speed) {
        if (trajectory.empty()) {
            return {0.0, 0.0, 0.0};
        }
        
        auto [steering, throttle, brake] = optimizeControl(state, trajectory, target_speed);
        
        return {steering, throttle, brake};
    }
  
    void setCostWeights(double w_cte, double w_epsi, double w_v, double w_delta, double w_a, double w_delta_d, double w_a_d) {
        weights.w_cte = w_cte;
        weights.w_epsi = w_epsi;
        weights.w_v = w_v;
        weights.w_delta = w_delta;
        weights.w_a = w_a;
        weights.w_delta_d = w_delta_d;
        weights.w_a_d = w_a_d;
    }
    
    void setHorizon(int N, int Nc = -1) {
        prediction_horizon = N;
        control_horizon = (Nc > 0) ? Nc : N;
    }
    
    void reset() {
        prev_steering = 0.0;
        prev_acceleration = 0.0;
    }
    
    //for debuging
    std::vector<VehicleState> getPredictedTrajectory(const VehicleState& initial_state, const std::vector<double>& steering_seq, const std::vector<double>& throttle_seq, const std::vector<double>& brake_seq) {
        
        std::vector<VehicleState> trajectory;
        VehicleState current_state = initial_state;
        trajectory.push_back(current_state);
        
        for (int i = 0; i < prediction_horizon && i < steering_seq.size(); ++i) {
            current_state = predictState(current_state, throttle_seq[i], brake_seq[i], steering_seq[i], dt);

            trajectory.push_back(current_state);
        }
        
        return trajectory;
    }
};

