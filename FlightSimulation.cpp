#include "FlightSimulation.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double DEG_TO_RAD = PI / 180.0;
constexpr double GRAVITY = 9.81;

}

FlightSimulation::FlightSimulation(
    const FlightSimulationConfig& config)
    : cfg(config),
      randomGenerator(42),
      normalDistribution(0.0, 1.0)
{
    reset();
}

double FlightSimulation::clamp(
    double value,
    double minimum,
    double maximum)
{
    return std::max(
        minimum,
        std::min(maximum, value));
}

double FlightSimulation::wrapAngle(double angle)
{
    while (angle > PI) {
        angle -= 2.0 * PI;
    }

    while (angle < -PI) {
        angle += 2.0 * PI;
    }

    return angle;
}

void FlightSimulation::reset()
{
    simulationTime = 0.0;
    trueState = cfg.initialTrueState;
    binsState = cfg.initialBinsState;

    output = FlightSimulationOutput();
    output.trueState = trueState;
    output.binsState = binsState;
    output.phase = "TAKEOFF";

    randomGenerator.seed(42);
}

void FlightSimulation::calculateControls(
    double& accelerationBodyX,
    double& accelerationBodyY,
    double& accelerationZ,
    double& angularVelocity,
    std::string& phase) const
{
    const double horizontalSpeed =
        std::sqrt(
            trueState.velocityX * trueState.velocityX +
            trueState.velocityY * trueState.velocityY);

    double targetSpeed = cfg.cruiseSpeed;
    double targetAltitude = cfg.cruiseAltitude;

    if (simulationTime < 10.0) {
        phase = "TAKEOFF";
    }
    else if (simulationTime < 55.0) {
        phase = "CRUISE";
    }
    else {
        phase = "DESCENT";
        targetAltitude = 0.0;

        if (simulationTime >= 68.0) {
            targetSpeed = 0.0;
        }
    }

    // Продольное ускорение разгоняет аппарат до заданной скорости.
    accelerationBodyX =
        clamp(
            0.8 * (targetSpeed - horizontalSpeed),
            -1.5,
            1.5);

    // Четыре поворота формируют замкнутый маршрут.
    const bool firstTurn =
        simulationTime >= 10.0 &&
        simulationTime < 16.0;

    const bool secondTurn =
        simulationTime >= 26.0 &&
        simulationTime < 32.0;

    const bool thirdTurn =
        simulationTime >= 42.0 &&
        simulationTime < 48.0;

    const bool fourthTurn =
        simulationTime >= 58.0 &&
        simulationTime < 64.0;

    angularVelocity = 0.0;

    if (firstTurn ||
        secondTurn ||
        thirdTurn ||
        fourthTurn) {
        angularVelocity = 15.0 * DEG_TO_RAD;
        phase = "TURN";
    }

    // Поперечное ускорение изменяет направление скорости при повороте.
    accelerationBodyY =
        horizontalSpeed * angularVelocity;

    // ПД-регулятор задаёт набор, удержание и снижение высоты.
    const double altitudeError =
        targetAltitude - trueState.z;

    accelerationZ =
        clamp(
            0.8 * altitudeError -
            1.3 * trueState.velocityZ,
            -2.0,
            2.0);
}

void FlightSimulation::simulateStep(double dt)
{
    double accelerationBodyX = 0.0;
    double accelerationBodyY = 0.0;
    double accelerationZ = 0.0;
    double angularVelocity = 0.0;
    std::string phase;

    calculateControls(
        accelerationBodyX,
        accelerationBodyY,
        accelerationZ,
        angularVelocity,
        phase);

    // Истинный курс: theta = theta0 + integral(omega dt).
    trueState.heading =
        wrapAngle(
            trueState.heading +
            angularVelocity * dt);

    const double trueCos =
        std::cos(trueState.heading);

    const double trueSin =
        std::sin(trueState.heading);

    // Переход от связанных ускорений к навигационным.
    const double trueAccelerationX =
        accelerationBodyX * trueCos -
        accelerationBodyY * trueSin;

    const double trueAccelerationY =
        accelerationBodyX * trueSin +
        accelerationBodyY * trueCos;

    // Интегрирование истинной скорости и координат.
    trueState.x +=
        trueState.velocityX * dt +
        0.5 * trueAccelerationX * dt * dt;

    trueState.y +=
        trueState.velocityY * dt +
        0.5 * trueAccelerationY * dt * dt;

    trueState.z +=
        trueState.velocityZ * dt +
        0.5 * accelerationZ * dt * dt;

    trueState.velocityX +=
        trueAccelerationX * dt;

    trueState.velocityY +=
        trueAccelerationY * dt;

    trueState.velocityZ +=
        accelerationZ * dt;

    if (trueState.z < 0.0) {
        trueState.z = 0.0;

        if (trueState.velocityZ < 0.0) {
            trueState.velocityZ = 0.0;
        }
    }

    const auto noise =
        [this]() {
            return normalDistribution(randomGenerator);
        };

    // Измерения содержат постоянное смещение и случайный шум.
    const double measuredAccelerationBodyX =
        accelerationBodyX +
        cfg.accelerometerBiasX +
        cfg.accelerometerNoiseStd * noise();

    const double measuredAccelerationBodyY =
        accelerationBodyY +
        cfg.accelerometerBiasY +
        cfg.accelerometerNoiseStd * noise();

    const double measuredAccelerationZ =
        accelerationZ +
        cfg.accelerometerBiasZ +
        cfg.accelerometerNoiseStd * noise();

    const double measuredAngularVelocity =
        angularVelocity +
        cfg.gyroscopeBias +
        cfg.gyroscopeNoiseStd * noise();

    // БИНС использует измеренную угловую скорость.
    binsState.heading =
        wrapAngle(
            binsState.heading +
            measuredAngularVelocity * dt);

    const double binsCos =
        std::cos(binsState.heading);

    const double binsSin =
        std::sin(binsState.heading);

    // Преобразование измеренных ускорений по оценённому курсу.
    const double binsAccelerationX =
        measuredAccelerationBodyX * binsCos -
        measuredAccelerationBodyY * binsSin;

    const double binsAccelerationY =
        measuredAccelerationBodyX * binsSin +
        measuredAccelerationBodyY * binsCos;

    // Интегрирование состояния БИНС.
    binsState.x +=
        binsState.velocityX * dt +
        0.5 * binsAccelerationX * dt * dt;

    binsState.y +=
        binsState.velocityY * dt +
        0.5 * binsAccelerationY * dt * dt;

    binsState.z +=
        binsState.velocityZ * dt +
        0.5 * measuredAccelerationZ * dt * dt;

    binsState.velocityX +=
        binsAccelerationX * dt;

    binsState.velocityY +=
        binsAccelerationY * dt;

    binsState.velocityZ +=
        measuredAccelerationZ * dt;

    if (binsState.z < 0.0) {
        binsState.z = 0.0;

        if (binsState.velocityZ < 0.0) {
            binsState.velocityZ = 0.0;
        }
    }

    output.trueState = trueState;
    output.binsState = binsState;

    output.trueAccelerationBodyX =
        accelerationBodyX;

    output.trueAccelerationBodyY =
        accelerationBodyY;

    output.trueAccelerationZ =
        accelerationZ;

    output.trueAngularVelocity =
        angularVelocity;

    // Стандартная IMU: X и Y — горизонтальные оси,
    // Z — вертикальная ось с удельной силой тяжести.
    output.measuredAcceleration =
        Vector3d(
            measuredAccelerationBodyX,
            measuredAccelerationBodyY,
            GRAVITY + measuredAccelerationZ);

    output.measuredAngularVelocity =
        Vector3d(
            0.0,
            0.0,
            measuredAngularVelocity);

    output.phase = phase;
}

FlightSimulationOutput FlightSimulation::update(double dt)
{
    if (dt <= 0.0) {
        return output;
    }

    double remainingTime = dt;

    while (remainingTime > 1.0e-12) {
        const double step =
            std::min(
                cfg.integrationStep,
                remainingTime);

        simulationTime += step;
        simulateStep(step);
        remainingTime -= step;
    }

    return output;
}

const SimulatedFlightState&
FlightSimulation::getTrueState() const
{
    return trueState;
}

const SimulatedFlightState&
FlightSimulation::getBinsState() const
{
    return binsState;
}

double FlightSimulation::getTime() const
{
    return simulationTime;
}
