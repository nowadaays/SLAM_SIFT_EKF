#pragma once

#include "INS.h"

// Измерения, которые должен предоставлять симулятор полёта.
struct FlightSimulationData {
    bool available = false;

    Vector3d acceleration;
    Vector3d angularVelocity;

    double altitude = 0.0;
    double verticalVelocity = 0.0;
};

// Возвращает измерения симулятора для указанного момента времени.
FlightSimulationData getFlightSimulationData(
    double time,
    double dt);