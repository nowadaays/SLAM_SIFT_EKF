#include "FlightSimulationProvider.h"

FlightSimulationData getFlightSimulationData(
    double time,
    double dt)
{
    (void)time;
    (void)dt;

    FlightSimulationData data;

    // Пока симулятор не подключён, данные не участвуют
    // в прогнозировании траектории.
    data.available = false;

    data.acceleration =
        Vector3d(0.0, 0.0, 9.81);

    data.angularVelocity =
        Vector3d(0.0, 0.0, 0.0);

    data.altitude = 0.0;
    data.verticalVelocity = 0.0;

    return data;
}