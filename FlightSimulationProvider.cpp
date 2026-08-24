#include "FlightSimulationProvider.h"

FlightSimulationData getFlightSimulationData(
    double time,
    double dt)
{
    static FlightSimulation simulation;
    static double previousTime = -1.0;

    if (previousTime >= 0.0 &&
        time < previousTime) {
        simulation.reset();
    }

    const FlightSimulationOutput simulationOutput =
        simulation.update(dt);

    previousTime = time;

    FlightSimulationData data;

    data.available = true;
    data.acceleration =
        simulationOutput.measuredAcceleration;

    data.angularVelocity =
        simulationOutput.measuredAngularVelocity;

    data.trueAcceleration =
        Vector3d(
            simulationOutput.trueAccelerationBodyX,
            simulationOutput.trueAccelerationBodyY,
            9.81 +
            simulationOutput.trueAccelerationZ);

    data.trueAngularVelocity =
        Vector3d(
            0.0,
            0.0,
            simulationOutput.trueAngularVelocity);

    data.altitude =
        simulationOutput.trueState.z;

    data.verticalVelocity =
        simulationOutput.trueState.velocityZ;

    data.trueState =
        simulationOutput.trueState;

    data.binsState =
        simulationOutput.binsState;

    data.phase =
        simulationOutput.phase;

    return data;
}