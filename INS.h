#pragma once
#include <cmath>
#include <tuple>

// ========== Вектор 3D ==========
struct Vector3d {
    double x, y, z;

    Vector3d() : x(0), y(0), z(0) {}
    Vector3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Операторы с векторами
    Vector3d operator+(const Vector3d& other) const {
        return Vector3d(x + other.x, y + other.y, z + other.z);
    }

    Vector3d operator-(const Vector3d& other) const {
        return Vector3d(x - other.x, y - other.y, z - other.z);
    }

    Vector3d operator-() const {
        return Vector3d(-x, -y, -z);
    }

    Vector3d operator*(double scalar) const {
        return Vector3d(x * scalar, y * scalar, z * scalar);
    }

    Vector3d operator/(double scalar) const {
        return Vector3d(x / scalar, y / scalar, z / scalar);
    }

    Vector3d& operator+=(const Vector3d& other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }

    Vector3d& operator-=(const Vector3d& other) {
        x -= other.x; y -= other.y; z -= other.z;
        return *this;
    }

    // Векторные операции
    Vector3d cross(const Vector3d& other) const {
        return Vector3d(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    double dot(const Vector3d& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    double norm() const {
        return sqrt(x * x + y * y + z * z);
    }

    void normalize() {
        double n = norm();
        if (n > 0) {
            x /= n; y /= n; z /= n;
        }
    }
};

// Скалярное умножение (скаляр * вектор)
inline Vector3d operator*(double scalar, const Vector3d& v) {
    return Vector3d(v.x * scalar, v.y * scalar, v.z * scalar);
}

// ========== Матрица 3x3 ==========
struct Matrix3d {
    double data[3][3];

    Matrix3d() {
        setIdentity();
    }

    Matrix3d(double diag) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                data[i][j] = (i == j) ? diag : 0.0;
    }

    // Умножение матрицы на вектор
    Vector3d operator*(const Vector3d& v) const {
        return Vector3d(
            data[0][0] * v.x + data[0][1] * v.y + data[0][2] * v.z,
            data[1][0] * v.x + data[1][1] * v.y + data[1][2] * v.z,
            data[2][0] * v.x + data[2][1] * v.y + data[2][2] * v.z
        );
    }

    // Умножение матриц
    Matrix3d operator*(const Matrix3d& other) const {
        Matrix3d result(0.0);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.data[i][j] = 0.0;
                for (int k = 0; k < 3; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    // Сложение матриц
    Matrix3d operator+(const Matrix3d& other) const {
        Matrix3d result;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    // Вычитание матриц
    Matrix3d operator-(const Matrix3d& other) const {
        Matrix3d result;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                result.data[i][j] = data[i][j] - other.data[i][j];
        return result;
    }

    // Транспонирование
    Matrix3d transpose() const {
        Matrix3d result;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                result.data[i][j] = data[j][i];
        return result;
    }

    // Установка единичной матрицы
    void setIdentity() {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                data[i][j] = (i == j) ? 1.0 : 0.0;
    }
};

// ========== Кватернион ==========
struct Quaterniond {
    double w, x, y, z;

    Quaterniond() : w(1), x(0), y(0), z(0) {}
    Quaterniond(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}

    void normalize() {
        double norm = sqrt(w * w + x * x + y * y + z * z);
        if (norm > 0) {
            w /= norm; x /= norm; y /= norm; z /= norm;
        }
    }

    // Преобразование в матрицу поворота
    Matrix3d toRotationMatrix() const {
        Matrix3d R;
        R.data[0][0] = 1 - 2 * (y * y + z * z);
        R.data[0][1] = 2 * (x * y + z * w);
        R.data[0][2] = 2 * (x * z - y * w);
        R.data[1][0] = 2 * (x * y - z * w);
        R.data[1][1] = 1 - 2 * (x * x + z * z);
        R.data[1][2] = 2 * (y * z + x * w);
        R.data[2][0] = 2 * (x * z + y * w);
        R.data[2][1] = 2 * (y * z - x * w);
        R.data[2][2] = 1 - 2 * (x * x + y * y);
        return R;
    }
};

// ========== Конфигурация INS ==========
struct INSConfig {
    Vector3d dw;        // смещение гироскопов
    Matrix3d kmw;       // масштабный коэффициент гироскопов
    Matrix3d fiw;       // нелинейность гироскопов
    Vector3d wnW;       // шумы гироскопов

    Vector3d da;        // смещение акселерометров
    Matrix3d kma;       // масштабный коэффициент акселерометров
    Matrix3d fia;       // нелинейность акселерометров
    Vector3d wnA;       // шумы акселерометров

    double Ue;          // угловая скорость Земли
    double a;           // большая полуось
    double b;           // малая полуось

    INSConfig() :
        dw(0, 0, 0),
        wnW(0, 0, 0),
        da(0, 0, 0),
        wnA(0, 0, 0),
        Ue(7.292115e-5),
        a(6378245.0),
        b(6356856.0) {
        kmw.setIdentity();
        fiw.setIdentity();
        kma.setIdentity();
        fia.setIdentity();
    }
};

// ========== Класс INS ==========
class INS {
public:
    INS(const INSConfig& config = INSConfig());

    void update(const Vector3d& A, const Vector3d& W, double dt,
        double time, double H_et, double DH_et);

    // Получение состояния
    void getNavParams(double(&np)[6]) const;
    void getPosition(double& lat, double& lon, double& alt) const;
    void getVelocity(double& vn, double& vh, double& ve) const;

    void resetPosition(double lat, double lon, double alt);
    void resetVelocity(double vn, double vh, double ve);

private:
    Vector3d DUS(const Vector3d& W, double time);
    Vector3d DLU(const Vector3d& A, double time);
    void Navigation(const Vector3d& A_dlu, const Matrix3d& Cbn,
        const double NP1[6], double dt, double H_et, double DH_et,
        double NP_new[6]);
    void Orientation(const Vector3d& W_dus, const Quaterniond& Q1,
        const Matrix3d& Cbn_prev, const double NP1[6],
        double dt, double H_et, double DH_et,
        Quaterniond& Q_new, Matrix3d& Cbn_new);

    std::tuple<Vector3d, double, double> EarthModel(double H, double FI, double LAMD) const;

    INSConfig cfg;
    double NP[6];        // [Vn, Vh, Ve, H, Fi, Lm]
    Quaterniond Q;
    Matrix3d Cbn;
};