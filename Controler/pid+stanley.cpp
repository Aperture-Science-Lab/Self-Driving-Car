#include <algorithm>
#include <tuple>
#include "model.cpp" 

class StanleyPIDController {
private:
    //stanley
    double k;
    //pid
    double kp_speed, ki_speed, kd_speed;
    double max_acceleration;
    double max_deceleration;
    double integral_error;
    double prev_error;
    //vehicle parameters
    double wheelbase;
    //output limits
    double max_steering_angle;
    double max_throttle;
    double max_brake;

public:
    StanleyPIDController(double stanley_gain = 2.5, double kp = 1.0, double ki = 0.1, double kd = 0.01, double max_accel = 3.0, double max_decel = 5.0, double wb = 2.7, double max_steer = 0.5236, double max_throttle_val = 1.0, double max_brake_val = 1.0)
        : k(stanley_gain), kp_speed(kp), ki_speed(ki), kd_speed(kd),
          max_acceleration(max_accel), max_deceleration(max_decel),
          integral_error(0), prev_error(0), wheelbase(wb),
          max_steering_angle(max_steer), max_throttle(max_throttle_val), max_brake(max_brake_val) {}


    double stanleyLateralControl(const VehicleState& state, const PathPoint& target) {
        double x = state.x;
        double y = state.y;
        double theta = state.theta;
        double v = state.v;

        double front_x = x + cos(theta) * wheelbase;
        double front_y = y + sin(theta) * wheelbase;

        // Path heading error
        double theta_e = theta - target.theta;
        
        theta_e = atan2(sin(theta_e), cos(theta_e));

        // Cross-track error
        double dx = front_x - target.x;
        double dy = front_y - target.y;
        double e_fa = -sin(target.theta) * dx + cos(target.theta) * dy;

        double delta = theta_e + atan2(k * e_fa, std::max(std::abs(v), 0.1));

        delta = std::max(-max_steering_angle, std::min(max_steering_angle, delta));
        
        return delta;
    }

    std::pair<double, double> pidLongitudinalControl(double current_speed, double target_speed, double dt) {
        double speed_error = target_speed - current_speed;
        
        double p_term = kp_speed * speed_error;
        
        integral_error += speed_error * dt;
        double i_term = ki_speed * integral_error;
        
        double derivative = (dt > 0) ? (speed_error - prev_error) / dt : 0.0;
        double d_term = kd_speed * derivative;

        double acceleration_cmd = p_term + i_term + d_term;
        acceleration_cmd = std::max(-max_deceleration, std::min(max_acceleration, acceleration_cmd));
        
        double throttle = 0.0;
        double brake = 0.0;
        
        if (acceleration_cmd > 0) {
            throttle = std::min(acceleration_cmd / max_acceleration * max_throttle, max_throttle);
        } else {
            brake = std::min(abs(acceleration_cmd) / max_deceleration * max_brake, max_brake);
        }

        // Anti-windup for integral term
        if ((acceleration_cmd >= max_acceleration && speed_error > 0) ||
            (acceleration_cmd <= -max_deceleration && speed_error < 0)) {
            integral_error -= speed_error * dt;
        }
        
        prev_error = speed_error;
        
        return {throttle, brake};
    }
    
    std::tuple<double, double, double> computeControl(const VehicleState& state, const PathPoint& target_point, double target_speed, double dt) {

        double steering = stanleyLateralControl(state, target_point);

        auto [throttle, brake] = pidLongitudinalControl(state.v, target_speed, dt);
        
        return {steering, throttle, brake};
    }

    void reset() {
        integral_error = 0.0;
        prev_error = 0.0;
    }

    void setStanleyGain(double new_k) { 
        k = new_k; 
    }

    void setPIDGains(double kp, double ki, double kd) {
        kp_speed = kp;
        ki_speed = ki;
        kd_speed = kd;
    }
    
    void setAccelerationLimits(double max_accel, double max_decel) {
        max_acceleration = max_accel;
        max_deceleration = max_decel;
    }
};

    