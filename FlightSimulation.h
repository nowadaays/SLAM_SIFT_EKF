#pragma once

#include "INS.h"

#include <random>
#include <string>

// Состояние в навигационной системе координат:
// X — север, Y — восток, Z — высота.
struct SimulatedFlightState {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    double velocityX = 0.0;
    double velocityY = 0.0;
    double velocityZ = 0.0;

    double heading = 0.0;
};

struct FlightSimulationConfig {
    SimulatedFlightState initialTrueState;
    SimulatedFlightState initialBinsState;

    double cruiseSpeed = 8.0;
    double cruiseAltitude = 30.0;

    double accelerometerBiasX = 0.015;
    double accelerometerBiasY = -0.012;
    double accelerometerBiasZ = 0.010;
    double gyroscopeBias = 0.0005;

    double accelerometerNoiseStd = 0.025;
    double gyroscopeNoiseStd = 0.001;

    // Внутренний шаг меньше шага видео для устойчивой интеграции.
    double integrationStep = 0.01;
};

struct FlightSimulationOutput {
    SimulatedFlightState trueState;
    SimulatedFlightState binsState;

    Vector3d measuredAcceleration;
    Vector3d measuredAngularVelocity;

    double trueAccelerationBodyX = 0.0;
    double trueAccelerationBodyY = 0.0;
    double trueAccelerationZ = 0.0;
    double trueAngularVelocity = 0.0;

    std::string phase;
};

class FlightSimulation {
public:
    explicit FlightSimulation(
        const FlightSimulationConfig& config =
            FlightSimulationConfig());

    void reset();

    FlightSimulationOutput update(double dt);

    const SimulatedFlightState& getTrueState() const;
    const SimulatedFlightState& getBinsState() const;

    double getTime() const;

private:
    static double clamp(
        double value,
        double minimum,
        double maximum);

    static double wrapAngle(double angle);

    void calculateControls(
        double& accelerationBodyX,
        double& accelerationBodyY,
        double& accelerationZ,
        double& angularVelocity,
        std::string& phase) const;

    void simulateStep(double dt);

    FlightSimulationConfig cfg;

    SimulatedFlightState trueState;
    SimulatedFlightState binsState;
    FlightSimulationOutput output;

    double simulationTime = 0.0;

    std::mt19937 randomGenerator;
    std::normal_distribution<double> normalDistribution;
};
