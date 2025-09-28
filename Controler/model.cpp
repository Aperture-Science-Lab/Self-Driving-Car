#include <cmath>
#include <algorithm>

struct VehicleState {
    double x;      // Global X position [m]
    double y;      // Global Y position [m]
    double theta;  // Heading angle [rad]
    double v;      // Longitudinal velocity [m/s]

    VehicleState(double x_=0.0, double y_=0.0, double theta_=0.0, double v_=0.0)
        : x(x_), y(y_), theta(theta_), v(v_) {}
};

struct PathPoint {
    double x;      // Global X position of the path point [m]
    double y;      // Global Y position of the path point [m]
    double theta;  // Heading angle of the path point [rad]
    double curvature; //for MPC [1/m]

    PathPoint(double x_=0.0, double y_=0.0, double theta_=0.0, double curvature_=0.0)
        : x(x_), y(y_), theta(theta_), curvature(curvature_) {}
};

class KinematicBicycleModel {
private:
    VehicleState state;

public:

    // Vehicle parameters
    const double wheelbase;
    const double max_steering_angle;
    const double max_velocity;
    const double min_velocity;
    const double max_acceleration;
    const double min_acceleration;

    KinematicBicycleModel(double wb, double max_steer = 0.5236, double max_vel = 50.0, double min_vel = 0.0, double max_accel = 3.0, double min_accel = -5.0, const VehicleState& initial_state = VehicleState())
        : wheelbase(wb), max_steering_angle(max_steer), max_velocity(max_vel), min_velocity(min_vel), max_acceleration(max_accel), min_acceleration(min_accel), state(initial_state) {}

    VehicleState getState() const {
        return state;
    }

    double getX() const { return state.x; }
    double getY() const { return state.y; }
    double getTheta() const { return state.theta; }
    double getVelocity() const { return state.v; }
    double getWheelbase() const { return wheelbase; }
    double getMaxSteeringAngle() const { return max_steering_angle; }
    double getMaxVelocity() const { return max_velocity; }
    double getMinVelocity() const { return min_velocity; }

    void setState(const VehicleState& new_state) {
        state = new_state;
    }

    void setPosition(double x, double y) { 
        state.x = x; 
        state.y = y; 
    }
    
    void setHeading(double theta) { 
        state.theta = theta; 
    }
    
    void setVelocity(double v) { 
        state.v = std::max(min_velocity, std::min(max_velocity, v)); 
    }

    void update(double throttle, double brake, double steering_angle, double dt) {
        // Clamp steering angle
        steering_angle = std::max(-max_steering_angle, std::min(max_steering_angle, steering_angle));

        state.x += state.v * std::cos(state.theta) * dt;
        state.y += state.v * std::sin(state.theta) * dt;
        
        state.theta += (state.v / wheelbase) * std::tan(steering_angle) * dt;
        
        while (state.theta > M_PI) state.theta -= 2 * M_PI;
        while (state.theta < -M_PI) state.theta += 2 * M_PI;
        
        double acceleration = 0.0;
        if (throttle > 0) {
            acceleration = throttle * max_acceleration;
        } else if (brake > 0) {
            acceleration = -brake * -1 * min_acceleration;
        }

        state.v = std::max(min_velocity, std::min(max_velocity, state.v));
    }

    void reset(const VehicleState& initial_state = VehicleState()) {
        state = initial_state;
    }
};