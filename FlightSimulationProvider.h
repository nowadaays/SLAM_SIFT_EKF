#pragma once

#include "FlightSimulation.h"

// Data passed from the simulator to INS and the user interface.
struct FlightSimulationData {
    bool available = false;

    Vector3d acceleration;
    Vector3d angularVelocity;

    Vector3d trueAcceleration;
    Vector3d trueAngularVelocity;

    double altitude = 0.0;
    double verticalVelocity = 0.0;

    SimulatedFlightState trueState;
    SimulatedFlightState binsState;

    std::string phase;
};

// Advances the model by dt and returns its current state.
FlightSimulationData getFlightSimulationData(
    double time,
    double dt);