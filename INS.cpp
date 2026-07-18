#include "INS.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

INS::INS(const INSConfig& config) : cfg(config) {
    // Начальные условия
    NP[0] = 0.0;   // Vn
    NP[1] = 0.0;   // Vh
    NP[2] = 0.0;   // Ve
    NP[3] = 0.0;   // H (высота)
    NP[4] = 55.7558 * M_PI / 180.0;  // Fi (широта) - Москва
    NP[5] = 37.6176 * M_PI / 180.0;  // Lm (долгота) - Москва

    Q = Quaterniond(1, 0, 0, 0);
    Cbn.setIdentity();
}

std::tuple<Vector3d, double, double> INS::EarthModel(double H, double FI, double LAMD) const {
    double e2 = 1.0 - (cfg.b * cfg.b) / (cfg.a * cfg.a);
    double W_ = sqrt(1.0 - e2 * sin(FI) * sin(FI));
    double Rn = cfg.a * (1.0 - e2) / pow(W_, 3);
    double R1 = cfg.a / W_;

    double g0_eq = 9.7803253359;
    double g = g0_eq * (1.0 + 0.001931853 * sin(FI) * sin(FI)) /
        sqrt(1.0 - e2 * sin(FI) * sin(FI));
    double Gg_value = g * (1.0 - 2.0 * H / cfg.a);

    Vector3d gravity(0.0, 0.0, Gg_value);
    return { gravity, R1, Rn };
}

Vector3d INS::DUS(const Vector3d& W, double time) {
    Vector3d Wp = cfg.fiw * W;
    Vector3d W_dus = (cfg.kmw * Wp) + cfg.dw + cfg.wnW;
    return W_dus;
}

Vector3d INS::DLU(const Vector3d& A, double time) {
    Vector3d Ap = cfg.fia * A;
    Vector3d A_dlu = (cfg.kma * Ap) + cfg.da + cfg.wnA;
    return A_dlu;
}

void INS::Navigation(const Vector3d& A_dlu, const Matrix3d& Cbn,
    const double NP1[6], double dt, double H_et, double DH_et,
    double NP_new[6]) {
    Vector3d VBE_n(NP1[0], NP1[1], NP1[2]);
    double H = H_et;
    double FI = NP1[4];
    double LAMD = NP1[5];

    Vector3d WEI_n(cfg.Ue * cos(FI), cfg.Ue * sin(FI), 0.0);

    // 
    Vector3d Gg;
    double R1, R2;
    std::tie(Gg, R1, R2) = EarthModel(H, FI, LAMD);

    double e2 = 1.0 - (cfg.b * cfg.b) / (cfg.a * cfg.a);
    Vector3d rn(-R1 * e2 * sin(FI) * cos(FI),
        R1 * (1.0 - e2 * sin(FI) * sin(FI)),
        0.0);

    Vector3d APn = -WEI_n.cross(WEI_n.cross(rn));
    Vector3d AKn = -2.0 * WEI_n.cross(VBE_n);

    Vector3d WGE_n(VBE_n.z / R1, VBE_n.z * tan(FI) / R1, -VBE_n.x / R2);
    Vector3d ATn = -WGE_n.cross(VBE_n);

    Vector3d A = Cbn * A_dlu;

    Vector3d DVBE_n;
    DVBE_n.x = A.x + APn.x + AKn.x + ATn.x + Gg.x;
    DVBE_n.y = A.y + APn.y + AKn.y + ATn.y + Gg.y;
    DVBE_n.z = A.z + APn.z + AKn.z + ATn.z + Gg.z;

    double DH = VBE_n.y;
    double DLAMD = VBE_n.z / (R1 * cos(FI));
    double DFI = VBE_n.x / R2;

    // Интегрирование методом Эйлера
    NP_new[0] = NP1[0] + DVBE_n.x * dt;
    NP_new[1] = NP1[1] + DVBE_n.y * dt;
    NP_new[2] = NP1[2] + DVBE_n.z * dt;
    NP_new[3] = NP1[3] + DH * dt;
    NP_new[4] = NP1[4] + DFI * dt;
    NP_new[5] = NP1[5] + DLAMD * dt;

    // Принудительная коррекция
    NP_new[3] = H_et;
    NP_new[1] = DH_et;
}

void INS::Orientation(const Vector3d& W_dus, const Quaterniond& Q1,
    const Matrix3d& Cbn_prev, const double NP1[6],
    double dt, double H_et, double DH_et,
    Quaterniond& Q_new, Matrix3d& Cbn_new) {
    Vector3d VBE_n(NP1[0], DH_et, NP1[2]);
    double H = H_et;
    double FI = NP1[4];
    double LAMD = NP1[5];

    Vector3d WEI_n(cfg.Ue * cos(FI), cfg.Ue * sin(FI), 0.0);

    // Разворачиваем tuple правильно
    Vector3d Gg;
    double R1, R2;
    std::tie(Gg, R1, R2) = EarthModel(H, FI, LAMD);

    Vector3d WNE_n(VBE_n.z / R1, VBE_n.z * tan(FI) / R1, -VBE_n.x / R2);
    Vector3d WNI_n;
    WNI_n.x = WNE_n.x + WEI_n.x;
    WNI_n.y = WNE_n.y + WEI_n.y;
    WNI_n.z = WNE_n.z + WEI_n.z;

    Matrix3d Anb = Cbn_prev.transpose();
    Vector3d WNI_b = Anb * WNI_n;

    Vector3d WBNb;
    WBNb.x = W_dus.x - WNI_b.x;
    WBNb.y = W_dus.y - WNI_b.y;
    WBNb.z = W_dus.z - WNI_b.z;

    double q0 = Q1.w, q1 = Q1.x, q2 = Q1.y, q3 = Q1.z;
    double QQQ = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;

    double DQ0 = 0.5 * (-q1 * WBNb.x - q2 * WBNb.y - q3 * WBNb.z + q0 * (1.0 - QQQ));
    double DQ1 = 0.5 * (q0 * WBNb.x - q3 * WBNb.y + q2 * WBNb.z + q1 * (1.0 - QQQ));
    double DQ2 = 0.5 * (q3 * WBNb.x + q0 * WBNb.y - q1 * WBNb.z + q2 * (1.0 - QQQ));
    double DQ3 = 0.5 * (-q2 * WBNb.x + q1 * WBNb.y + q0 * WBNb.z + q3 * (1.0 - QQQ));

    Q_new.w = q0 + DQ0 * dt;
    Q_new.x = q1 + DQ1 * dt;
    Q_new.y = q2 + DQ2 * dt;
    Q_new.z = q3 + DQ3 * dt;
    Q_new.normalize();

    Cbn_new = Q_new.toRotationMatrix();
}

void INS::update(const Vector3d& A, const Vector3d& W, double dt,
    double time, double H_et, double DH_et) {
    Vector3d W_dus = DUS(W, time);
    Vector3d A_dlu = DLU(A, time);

    double NP_new[6];
    Navigation(A_dlu, Cbn, NP, dt, H_et, DH_et, NP_new);

    Quaterniond Q_new;
    Matrix3d Cbn_new;
    Orientation(W_dus, Q, Cbn, NP, dt, H_et, DH_et, Q_new, Cbn_new);

    // Обновление состояния
    for (int i = 0; i < 6; i++) NP[i] = NP_new[i];
    Q = Q_new;
    Cbn = Cbn_new;
}

void INS::getNavParams(double(&np)[6]) const {
    for (int i = 0; i < 6; i++) np[i] = NP[i];
}

void INS::getPosition(double& lat, double& lon, double& alt) const {
    lat = NP[4] * 180.0 / M_PI;  // в градусы
    lon = NP[5] * 180.0 / M_PI;
    alt = NP[3];
}

void INS::getVelocity(double& vn, double& vh, double& ve) const {
    vn = NP[0];
    vh = NP[1];
    ve = NP[2];
}

void INS::resetPosition(double lat, double lon, double alt) {
    NP[4] = lat * M_PI / 180.0;
    NP[5] = lon * M_PI / 180.0;
    NP[3] = alt;
}

void INS::resetVelocity(double vn, double vh, double ve) {
    NP[0] = vn;
    NP[1] = vh;
    NP[2] = ve;
}