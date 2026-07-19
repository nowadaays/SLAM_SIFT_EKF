#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

#include "INS.h"

// ДОБАВЛЕНО:
// Отдельный класс расчёта траектории.
#include "TrajectoryCalculator.h"

using namespace cv;
using namespace std;

// ============================================================
// ПОЛНЫЕ ПУТИ К ФАЙЛАМ
// ============================================================
const string MAP_PATH =
"C:\\Diplom\\OpenCV_SIFT\\x64\\Debug\\map.jpg";

const string VIDEO_PATH =
"C:\\Diplom\\OpenCV_SIFT\\x64\\Debug\\video.mp4";

const string TRAJECTORY_PATH =
"C:\\Diplom\\OpenCV_SIFT\\x64\\Debug\\trajectory.txt";

// ============================================================
// ПАРАМЕТРЫ SIFT
// ============================================================
constexpr int SIFT_FEATURE_COUNT = 5000;
constexpr double SIFT_CONTRAST_THRESHOLD = 0.015;
constexpr double SIFT_EDGE_THRESHOLD = 15.0;

constexpr float LOWE_RATIO = 0.82f;

constexpr int MIN_MATCHES = 5;
constexpr int MIN_INLIERS = 4;
constexpr double MIN_INLIER_RATIO = 0.50;

// Минимальное распределение совпадений по кадру.
constexpr double MIN_POINT_SPREAD = 0.12;

// ============================================================
// ПАРАМЕТРЫ ОБЛАСТИ БИНС
// ============================================================
constexpr double BINS_INITIAL_RADIUS_PX = 160.0;
constexpr double BINS_MIN_RADIUS_PX = 100.0;
constexpr double BINS_MAX_RADIUS_PX = 700.0;
constexpr double BINS_RADIUS_GROWTH_PX_PER_SEC = 70.0;
constexpr double LOW_FEATURE_EXPANSION_PX = 10.0;

constexpr int MIN_MAP_KEYPOINTS = 15;

// ============================================================
// ДОБАВЛЕНО:
// ВХОДНЫЕ ДАННЫЕ БУДУЩЕЙ СИМУЛЯЦИИ
// ============================================================

struct FlightSimulationData {
    // false — симуляция пока отсутствует.
    // В таком состоянии данные не влияют на траекторию.
    bool available = false;

    Vector3d acceleration;
    Vector3d angularVelocity;

    double altitude = 0.0;
    double verticalVelocity = 0.0;
};

// ============================================================
// ДОБАВЛЕНО:
// ЗАГЛУШКА БУДУЩЕГО СИМУЛЯТОРА
//
// Сейчас функция возвращает available=false.
//
// Когда появится симулятор полёта, необходимо будет изменить
// только содержимое этой функции и передать:
//
// acceleration      — показания акселерометра;
// angularVelocity   — показания гироскопа;
// altitude          — высоту;
// verticalVelocity  — вертикальную скорость.
//
// Весь остальной расчёт траектории менять не потребуется.
// ============================================================
FlightSimulationData getFlightSimulationData(
    double time,
    double dt)
{
    (void)time;
    (void)dt;

    FlightSimulationData data;

    data.available = false;

    // Заглушки не используются, пока available == false.
    data.acceleration =
        Vector3d(0.0, 0.0, 9.81);

    data.angularVelocity =
        Vector3d(0.0, 0.0, 0.0);

    data.altitude = 0.0;
    data.verticalVelocity = 0.0;

    /*
    // ПРИМЕР БУДУЩЕГО ПОДКЛЮЧЕНИЯ:

    simulator.update(dt);

    data.available = true;

    data.acceleration =
        simulator.getAcceleration();

    data.angularVelocity =
        simulator.getAngularVelocity();

    data.altitude =
        simulator.getAltitude();

    data.verticalVelocity =
        simulator.getVerticalVelocity();
    */

    return data;
}

// ============================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================

bool pointInsideImage(
    const Point2f& point,
    const Mat& image)
{
    return std::isfinite(point.x) &&
        std::isfinite(point.y) &&
        point.x >= 0.0f &&
        point.x < static_cast<float>(image.cols) &&
        point.y >= 0.0f &&
        point.y < static_cast<float>(image.rows);
}

// Подготовка изображения с помощью CLAHE.
Mat preprocessImage(
    const Mat& gray,
    const Ptr<CLAHE>& clahe)
{
    Mat result;

    clahe->apply(gray, result);

    // Слабое размытие только для уменьшения шума.
    GaussianBlur(
        result,
        result,
        Size(3, 3),
        0.4);

    return result;
}

// Маленькие кадры увеличиваются перед SIFT.
// Координаты центра далее берутся уже в увеличенном кадре.
Mat upscaleSmallFrame(
    const Mat& frame,
    double& scale)
{
    int minimumSide =
        std::min(frame.cols, frame.rows);

    scale = 1.0;

    // Для кадров размером около 90x90 получится увеличение x4.
    if (minimumSide < 360) {
        scale =
            360.0 /
            static_cast<double>(minimumSide);

        scale = std::min(scale, 4.0);
    }

    if (scale <= 1.0) {
        return frame.clone();
    }

    Mat enlarged;

    resize(
        frame,
        enlarged,
        Size(),
        scale,
        scale,
        INTER_CUBIC);

    return enlarged;
}

// Взаимная проверка совпадений:
//
// точка карты должна выбрать точку кадра,
// а эта точка кадра должна выбрать ту же точку карты.
vector<DMatch> findMutualMatches(
    const Mat& mapDescriptors,
    const Mat& frameDescriptors,
    BFMatcher& matcher)
{
    vector<vector<DMatch>> mapToFrame;
    vector<vector<DMatch>> frameToMap;

    matcher.knnMatch(
        mapDescriptors,
        frameDescriptors,
        mapToFrame,
        2);

    matcher.knnMatch(
        frameDescriptors,
        mapDescriptors,
        frameToMap,
        2);

    vector<DMatch> matches;

    for (size_t mapIndex = 0;
        mapIndex < mapToFrame.size();
        mapIndex++) {

        const vector<DMatch>& forward =
            mapToFrame[mapIndex];

        if (forward.size() < 2) {
            continue;
        }

        if (forward[0].distance >=
            LOWE_RATIO * forward[1].distance) {
            continue;
        }

        const DMatch& bestForward =
            forward[0];

        int frameIndex =
            bestForward.trainIdx;

        if (frameIndex < 0 ||
            frameIndex >=
            static_cast<int>(
                frameToMap.size())) {
            continue;
        }

        const vector<DMatch>& reverse =
            frameToMap[frameIndex];

        if (reverse.size() < 2) {
            continue;
        }

        if (reverse[0].distance >=
            LOWE_RATIO * reverse[1].distance) {
            continue;
        }

        // reverse[0].trainIdx — индекс точки карты.
        if (reverse[0].trainIdx !=
            bestForward.queryIdx) {
            continue;
        }

        matches.push_back(bestForward);
    }

    sort(
        matches.begin(),
        matches.end(),
        [](const DMatch& left,
            const DMatch& right) {
                return left.distance <
                    right.distance;
        });

    return matches;
}

// Определение позиции центра кадра на карте.
bool estimateMapPosition(
    const vector<KeyPoint>& mapKeypoints,
    const vector<KeyPoint>& frameKeypoints,
    const vector<DMatch>& matches,
    const Size& frameSize,
    Point2f& mapPosition,
    int& inlierCount,
    double& inlierRatio,
    string& failureReason)
{
    inlierCount = 0;
    inlierRatio = 0.0;

    if (matches.size() <
        static_cast<size_t>(MIN_MATCHES)) {
        failureReason =
            "Not enough mutual matches";
        return false;
    }

    vector<Point2f> mapPoints;
    vector<Point2f> framePoints;

    mapPoints.reserve(matches.size());
    framePoints.reserve(matches.size());

    for (const DMatch& match : matches) {
        mapPoints.push_back(
            mapKeypoints[
                match.queryIdx].pt);

        framePoints.push_back(
            frameKeypoints[
                match.trainIdx].pt);
    }

    Mat inlierMask;

    // Для вертикальной камеры и плоской карты устойчивое
    // аффинное преобразование обычно точнее гомографии
    // при небольшом количестве точек.
    Mat affineTransform =
        estimateAffinePartial2D(
            framePoints,
            mapPoints,
            inlierMask,
            RANSAC,
            3.0,
            3000,
            0.995,
            10);

    if (affineTransform.empty() ||
        inlierMask.empty()) {
        failureReason =
            "Affine RANSAC failed";
        return false;
    }

    inlierCount =
        countNonZero(inlierMask);

    inlierRatio =
        static_cast<double>(inlierCount) /
        static_cast<double>(matches.size());

    if (inlierCount < MIN_INLIERS) {
        failureReason =
            "Not enough RANSAC inliers";
        return false;
    }

    if (inlierRatio <
        MIN_INLIER_RATIO) {
        failureReason =
            "Low RANSAC inlier ratio";
        return false;
    }

    // Проверяем, что подтверждённые точки распределены
    // по кадру, а не находятся в одном маленьком месте.
    vector<Point2f> inlierFramePoints;

    for (int i = 0;
        i < static_cast<int>(
            framePoints.size());
        i++) {

        if (inlierMask.at<uchar>(i) != 0) {
            inlierFramePoints.push_back(
                framePoints[i]);
        }
    }

    if (inlierFramePoints.size() <
        static_cast<size_t>(MIN_INLIERS)) {
        failureReason =
            "Invalid inlier points";
        return false;
    }

    Rect pointBounds =
        boundingRect(inlierFramePoints);

    double spreadX =
        static_cast<double>(
            pointBounds.width) /
        static_cast<double>(
            frameSize.width);

    double spreadY =
        static_cast<double>(
            pointBounds.height) /
        static_cast<double>(
            frameSize.height);

    if (spreadX < MIN_POINT_SPREAD ||
        spreadY < MIN_POINT_SPREAD) {
        failureReason =
            "RANSAC points are too clustered";
        return false;
    }

    vector<Point2f> frameCenter = {
        Point2f(
            frameSize.width * 0.5f,
            frameSize.height * 0.5f)
    };

    vector<Point2f> transformedCenter;

    transform(
        frameCenter,
        transformedCenter,
        affineTransform);

    if (transformedCenter.empty()) {
        failureReason =
            "Center transformation failed";
        return false;
    }

    mapPosition =
        transformedCenter[0];

    failureReason = "NONE";

    return std::isfinite(mapPosition.x) &&
        std::isfinite(mapPosition.y);
}

int main()
{
    cout << "========================================" << endl;
    cout << " Drone Navigation System (BINS + SIFT)" << endl;
    cout << "========================================" << endl;
    cout << endl;

    // ========================================================
    // ИНИЦИАЛИЗАЦИЯ БИНС
    // ========================================================
    INSConfig config;
    INS ins(config);

    // ========================================================
    // ДОБАВЛЕНО:
    // ИНИЦИАЛИЗАЦИЯ КАЛЬКУЛЯТОРА ТРАЕКТОРИИ
    // ========================================================
    TrajectoryConfig trajectoryConfig;

    // Временная заглушка масштаба.
    // После появления симуляции необходимо указать
    // реальное количество пикселей на один метр.
    trajectoryConfig.pixelsPerMeter = 3.0;

    trajectoryConfig.siftCorrectionGain = 0.75;
    trajectoryConfig.minimumPointDistance = 0.5;
    trajectoryConfig.maximumHistorySize = 300;

    TrajectoryCalculator trajectoryCalculator(
        trajectoryConfig);

    double dt = 0.033;
    double currentTime = 0.0;

    // Старые заглушки оставлены без изменений.
    Vector3d acceleration(0.0, 0.0, 9.81);
    Vector3d angularVelocity(0.0, 0.0, 0.0);

    // ========================================================
    // ЗАГРУЗКА КАРТЫ
    // ========================================================
    cout << "Loading map..." << endl;

    Mat mapOriginal =
        imread(
            MAP_PATH,
            IMREAD_GRAYSCALE);

    if (mapOriginal.empty()) {
        cout << "ERROR: Failed to load map!" << endl;
        cout << MAP_PATH << endl;
        return -1;
    }

    cout << "Map loaded. Size: "
        << mapOriginal.cols << "x"
        << mapOriginal.rows << endl;

    // ========================================================
    // ЗАГРУЗКА ВИДЕО
    // ========================================================
    cout << "Loading video..." << endl;

    VideoCapture cap(VIDEO_PATH);

    if (!cap.isOpened()) {
        cout << "ERROR: Failed to open video!" << endl;
        cout << VIDEO_PATH << endl;
        return -1;
    }

    double videoFPS =
        cap.get(CAP_PROP_FPS);

    if (videoFPS > 0.0) {
        dt = 1.0 / videoFPS;
    }

    cout << "Video loaded. FPS: "
        << videoFPS << endl;

    // ========================================================
    // SIFT И CLAHE
    // ========================================================
    Ptr<CLAHE> clahe =
        createCLAHE(
            2.0,
            Size(8, 8));

    Mat processedMap =
        preprocessImage(
            mapOriginal,
            clahe);

    Ptr<SIFT> sift =
        SIFT::create(
            SIFT_FEATURE_COUNT,
            3,
            SIFT_CONTRAST_THRESHOLD,
            SIFT_EDGE_THRESHOLD,
            1.6);

    BFMatcher matcher(NORM_L2);

    // ========================================================
    // ГЛОБАЛЬНЫЕ ТОЧКИ КАРТЫ
    //
    // Используются исключительно для начальной локализации.
    // После определения начальной позиции они больше
    // не участвуют в поиске.
    // ========================================================
    vector<KeyPoint> globalMapKeypoints;
    Mat globalMapDescriptors;

    sift->detectAndCompute(
        processedMap,
        noArray(),
        globalMapKeypoints,
        globalMapDescriptors);

    cout << "Global map keypoints: "
        << globalMapKeypoints.size()
        << endl;

    if (globalMapDescriptors.empty()) {
        cout << "ERROR: Map has no SIFT descriptors."
            << endl;
        return -1;
    }

    // ========================================================
    // ПЕРЕМЕННЫЕ НАВИГАЦИИ
    // ========================================================
    bool initialPositionFound = false;
    bool siftValid = false;

    int frameID = 0;
    int lostFrames = 0;

    Point2f siftPosition(0.0f, 0.0f);
    Point2f binsPosition(0.0f, 0.0f);
    Point2f fusedPosition(0.0f, 0.0f);

    double binsErrorRadius =
        BINS_INITIAL_RADIUS_PX;

    int goodMatchesCount = 0;
    int inlierCount = 0;
    double inlierRatio = 0.0;

    string failureReason =
        "Initial localization required";

    // Переменная сохранена, чтобы не менять
    // существующую визуализацию.
    vector<Point2f> trajectory;

    // ========================================================
    // СТАТИСТИКА
    // ========================================================
    double processedFrames = 0.0;
    double successfulFrames = 0.0;

    // ========================================================
    // ФАЙЛ ТРАЕКТОРИИ
    // ========================================================
    ofstream trajectoryFile(
        TRAJECTORY_PATH);

    if (!trajectoryFile.is_open()) {
        cout << "ERROR: Cannot create trajectory file."
            << endl;
        return -1;
    }

    // ДОБАВЛЕНО:
    // Новые поля итоговой траектории, скорости и коррекции.
    trajectoryFile
        << "frame\ttime\t"
        << "SIFT_x\tSIFT_y\t"
        << "BINS_x\tBINS_y\t"
        << "radius\tmatches\tinliers\t"
        << "inlier_ratio\tvalid\tmode\t"
        << "Fused_x\tFused_y\t"
        << "motion_available\t"
        << "velocity_n\tvelocity_e\tvelocity_u\t"
        << "distance_px\tcorrection_px\n";

    cout << endl;
    cout << "Starting processing..." << endl;
    cout << endl;

    // ========================================================
    // ГЛАВНЫЙ ЦИКЛ
    // ========================================================
    while (true) {
        Mat frame;

        cap >> frame;

        if (frame.empty()) {
            cout << "End of video." << endl;
            break;
        }

        frameID++;
        processedFrames++;
        currentTime += dt;

        // ====================================================
        // ДОБАВЛЕНО:
        // ПОЛУЧЕНИЕ ДАННЫХ БУДУЩЕЙ СИМУЛЯЦИИ
        // ====================================================
        FlightSimulationData simulationData =
            getFlightSimulationData(
                currentTime,
                dt);

        // Пока симуляции нет, INS получает те же данные,
        // которые использовались в рабочей версии программы.
        Vector3d currentAcceleration =
            simulationData.available
            ? simulationData.acceleration
            : acceleration;

        Vector3d currentAngularVelocity =
            simulationData.available
            ? simulationData.angularVelocity
            : angularVelocity;

        double currentAltitude =
            simulationData.available
            ? simulationData.altitude
            : 0.0;

        double currentVerticalVelocity =
            simulationData.available
            ? simulationData.verticalVelocity
            : 0.0;

        // Обновление математической модели БИНС.
        // При отсутствии симуляции сохранено старое поведение.
        ins.update(
            currentAcceleration,
            currentAngularVelocity,
            dt,
            currentTime,
            currentAltitude,
            currentVerticalVelocity);

        // ====================================================
        // ДОБАВЛЕНО:
        // ПОЛУЧЕНИЕ СКОРОСТИ БИНС
        // ====================================================
        double binsVelocityNorth = 0.0;
        double binsVelocityUp = 0.0;
        double binsVelocityEast = 0.0;

        ins.getVelocity(
            binsVelocityNorth,
            binsVelocityUp,
            binsVelocityEast);

        // ====================================================
        // ДОБАВЛЕНО:
        // ПРОГНОЗ ТРАЕКТОРИИ ПО БИНС
        // ====================================================
        TrajectoryMotionData motionData;

        // Пока симуляции нет, available=false,
        // поэтому предсказание не изменяет позицию.
        motionData.available =
            simulationData.available;

        motionData.velocityNorth =
            binsVelocityNorth;

        motionData.velocityUp =
            binsVelocityUp;

        motionData.velocityEast =
            binsVelocityEast;

        motionData.dt = dt;

        trajectoryCalculator.predict(
            motionData);

        // После появления симуляции binsPosition будет
        // перемещаться по скорости БИНС.
        if (trajectoryCalculator.isInitialized()) {
            binsPosition =
                trajectoryCalculator.getBinsPosition();

            fusedPosition =
                trajectoryCalculator.getFusedPosition();
        }

        // ====================================================
        // ПОДГОТОВКА КАДРА
        // ====================================================
        Mat gray;

        cvtColor(
            frame,
            gray,
            COLOR_BGR2GRAY);

        double frameScale = 1.0;

        Mat enlargedFrame =
            upscaleSmallFrame(
                gray,
                frameScale);

        Mat processedFrame =
            preprocessImage(
                enlargedFrame,
                clahe);

        vector<KeyPoint> frameKeypoints;
        Mat frameDescriptors;

        sift->detectAndCompute(
            processedFrame,
            noArray(),
            frameKeypoints,
            frameDescriptors);

        // ====================================================
        // ВЫБОР ОБЛАСТИ ПОИСКА
        // ====================================================
        bool globalInitializationMode =
            !initialPositionFound;

        vector<KeyPoint> activeMapKeypoints;
        Mat activeMapDescriptors;

        Mat binsMask =
            Mat::zeros(
                processedMap.size(),
                CV_8UC1);

        double currentSearchRadius =
            binsErrorRadius;

        if (globalInitializationMode) {
            // Глобальный поиск разрешён только до первой
            // надёжной локализации.
            activeMapKeypoints =
                globalMapKeypoints;

            activeMapDescriptors =
                globalMapDescriptors;
        }
        else {
            // После инициализации поиск карты выполняется
            // строго внутри области ошибки БИНС.
            circle(
                binsMask,
                binsPosition,
                cvRound(currentSearchRadius),
                Scalar(255),
                FILLED,
                LINE_AA);

            sift->detectAndCompute(
                processedMap,
                binsMask,
                activeMapKeypoints,
                activeMapDescriptors);
        }

        siftValid = false;
        goodMatchesCount = 0;
        inlierCount = 0;
        inlierRatio = 0.0;

        failureReason = "Unknown";

        vector<DMatch> goodMatches;

        // ====================================================
        // СОПОСТАВЛЕНИЕ
        // ====================================================
        if (frameDescriptors.empty()) {
            failureReason =
                "Frame descriptors are empty";
        }
        else if (activeMapDescriptors.empty()) {
            failureReason =
                "Map descriptors are empty";
        }
        else {
            goodMatches =
                findMutualMatches(
                    activeMapDescriptors,
                    frameDescriptors,
                    matcher);

            goodMatchesCount =
                static_cast<int>(
                    goodMatches.size());

            Point2f candidatePosition;

            bool positionEstimated =
                estimateMapPosition(
                    activeMapKeypoints,
                    frameKeypoints,
                    goodMatches,
                    processedFrame.size(),
                    candidatePosition,
                    inlierCount,
                    inlierRatio,
                    failureReason);

            if (positionEstimated &&
                !pointInsideImage(
                    candidatePosition,
                    processedMap)) {

                positionEstimated = false;

                failureReason =
                    "Position outside map";
            }

            // После инициализации позиция также обязана
            // находиться внутри области ошибки БИНС.
            if (positionEstimated &&
                initialPositionFound) {

                double distanceToBINS =
                    norm(
                        candidatePosition -
                        binsPosition);

                if (distanceToBINS >
                    currentSearchRadius) {

                    positionEstimated = false;

                    failureReason =
                        "Position outside BINS area";
                }
            }

            if (positionEstimated) {
                siftPosition =
                    candidatePosition;

                // ============================================
                // ДОБАВЛЕНО:
                // КОРРЕКЦИЯ ТРАЕКТОРИИ ПО SIFT
                //
                // Без симуляции результат полностью совпадает
                // с candidatePosition, как в рабочей версии.
                // ============================================
                trajectoryCalculator.correct(
                    true,
                    siftPosition);

                binsPosition =
                    trajectoryCalculator.getBinsPosition();

                fusedPosition =
                    trajectoryCalculator.getFusedPosition();

                siftValid = true;
                successfulFrames++;
                lostFrames = 0;

                binsErrorRadius =
                    BINS_MIN_RADIUS_PX;

                if (!initialPositionFound) {
                    initialPositionFound = true;

                    cout << endl;

                    cout << "INITIAL POSITION FOUND: ("
                        << siftPosition.x << ", "
                        << siftPosition.y << ")"
                        << endl;

                    cout << "Matches: "
                        << goodMatchesCount
                        << ", inliers: "
                        << inlierCount
                        << endl;

                    cout << endl;
                }
            }
        }

        // ====================================================
        // ДОБАВЛЕНО:
        // ОБРАБОТКА SIFT LOST В КАЛЬКУЛЯТОРЕ ТРАЕКТОРИИ
        // ====================================================
        if (!siftValid) {
            trajectoryCalculator.correct(
                false,
                Point2f(0.0f, 0.0f));

            if (trajectoryCalculator.isInitialized()) {
                binsPosition =
                    trajectoryCalculator.getBinsPosition();

                fusedPosition =
                    trajectoryCalculator.getFusedPosition();
            }
        }

        // ====================================================
        // РОСТ ОБЛАСТИ ПРИ ПОТЕРЕ SIFT
        // ====================================================
        if (!siftValid) {
            lostFrames++;

            if (initialPositionFound) {
                binsErrorRadius +=
                    BINS_RADIUS_GROWTH_PX_PER_SEC *
                    dt;

                if (activeMapKeypoints.size() <
                    MIN_MAP_KEYPOINTS) {
                    binsErrorRadius +=
                        LOW_FEATURE_EXPANSION_PX;
                }

                binsErrorRadius =
                    std::min(
                        BINS_MAX_RADIUS_PX,
                        binsErrorRadius);
            }
        }

        // ====================================================
        // ТРАЕКТОРИЯ
        // ====================================================
        // ДОБАВЛЕНО:
        // Получаем уже рассчитанную объединённую траекторию
        // из отдельного класса.
        //
        // Переменная trajectory сохранена для совместимости
        // с существующим кодом визуализации.
        trajectory =
            trajectoryCalculator.getFusedTrajectory();

        // ====================================================
        // КОНСОЛЬНЫЙ ВЫВОД
        // ====================================================
        string mode =
            globalInitializationMode
            ? "INITIAL_GLOBAL_SEARCH"
            : "BINS_AREA_ONLY";

        cout << "----------------------------------------"
            << endl;

        cout << "Frame:             "
            << frameID << endl;

        cout << "Mode:              "
            << mode << endl;

        cout << "SIFT:              "
            << (siftValid
                ? "ACTIVE"
                : "LOST")
            << endl;

        cout << "Reason:            "
            << failureReason << endl;

        cout << "Frame size:        "
            << gray.cols << "x"
            << gray.rows << endl;

        cout << "SIFT frame size:   "
            << processedFrame.cols << "x"
            << processedFrame.rows << endl;

        cout << "Frame keypoints:   "
            << frameKeypoints.size() << endl;

        cout << "Map keypoints:     "
            << activeMapKeypoints.size() << endl;

        cout << "Mutual matches:    "
            << goodMatchesCount << endl;

        cout << "RANSAC inliers:    "
            << inlierCount << endl;

        cout << "Inlier ratio:      "
            << fixed << setprecision(2)
            << inlierRatio << endl;

        cout << "BINS radius:       "
            << binsErrorRadius << " px"
            << endl;

        // ====================================================
        // ДОБАВЛЕНО:
        // ИНФОРМАЦИЯ О ТРАЕКТОРИИ
        // ====================================================
        cout << "Motion data:       "
            << (simulationData.available
                ? "AVAILABLE"
                : "STUB / NOT AVAILABLE")
            << endl;

        cout << "BINS velocity N:   "
            << binsVelocityNorth
            << " m/s" << endl;

        cout << "BINS velocity E:   "
            << binsVelocityEast
            << " m/s" << endl;

        cout << "BINS velocity U:   "
            << binsVelocityUp
            << " m/s" << endl;

        cout << "Flight distance:   "
            << fixed << setprecision(1)
            << trajectoryCalculator.getDistancePixels()
            << " px"
            << endl;

        cout << "SIFT correction:   "
            << trajectoryCalculator
            .getLastCorrectionPixels()
            << " px"
            << endl;

        if (initialPositionFound) {
            cout << "Position:          ("
                << cvRound(fusedPosition.x)
                << ", "
                << cvRound(fusedPosition.y)
                << ")"
                << endl;
        }

        cout << endl;

        // ====================================================
        // ЗАПИСЬ ТРАЕКТОРИИ
        // ====================================================
        trajectoryFile
            << frameID << "\t"
            << currentTime << "\t"

            << (siftValid
                ? siftPosition.x
                : -1.0f) << "\t"

            << (siftValid
                ? siftPosition.y
                : -1.0f) << "\t"

            << (initialPositionFound
                ? binsPosition.x
                : -1.0f) << "\t"

            << (initialPositionFound
                ? binsPosition.y
                : -1.0f) << "\t"

            << binsErrorRadius << "\t"
            << goodMatchesCount << "\t"
            << inlierCount << "\t"
            << inlierRatio << "\t"
            << siftValid << "\t"
            << mode << "\t"

            // ДОБАВЛЕНО:
            << (trajectoryCalculator.isInitialized()
                ? fusedPosition.x
                : -1.0f) << "\t"

            << (trajectoryCalculator.isInitialized()
                ? fusedPosition.y
                : -1.0f) << "\t"

            << simulationData.available << "\t"
            << binsVelocityNorth << "\t"
            << binsVelocityEast << "\t"
            << binsVelocityUp << "\t"

            << trajectoryCalculator
            .getDistancePixels() << "\t"

            << trajectoryCalculator
            .getLastCorrectionPixels()

            << "\n";

        // ====================================================
        // ВИЗУАЛИЗАЦИЯ
        // ====================================================
        Mat mapColor;
        Mat frameColor;

        cvtColor(
            processedMap,
            mapColor,
            COLOR_GRAY2BGR);

        cvtColor(
            processedFrame,
            frameColor,
            COLOR_GRAY2BGR);

        // Область ошибки показывается только после
        // начальной глобальной локализации.
        if (initialPositionFound) {
            Mat overlay =
                mapColor.clone();

            circle(
                overlay,
                binsPosition,
                cvRound(currentSearchRadius),
                Scalar(0, 0, 255),
                FILLED,
                LINE_AA);

            addWeighted(
                overlay,
                0.22,
                mapColor,
                0.78,
                0.0,
                mapColor);

            circle(
                mapColor,
                binsPosition,
                cvRound(currentSearchRadius),
                Scalar(0, 0, 255),
                2,
                LINE_AA);
        }

        int mapWidth =
            mapColor.cols;

        int frameWidth =
            frameColor.cols;

        int contentHeight =
            std::max(
                mapColor.rows,
                frameColor.rows);

        constexpr int PANEL_HEIGHT = 140;

        Mat result = Mat::zeros(
            Size(
                mapWidth + frameWidth,
                contentHeight +
                PANEL_HEIGHT),
            CV_8UC3);

        mapColor.copyTo(
            result(
                Rect(
                    0,
                    0,
                    mapColor.cols,
                    mapColor.rows)));

        frameColor.copyTo(
            result(
                Rect(
                    mapWidth,
                    0,
                    frameColor.cols,
                    frameColor.rows)));

        // Линии совпадений.
        size_t matchesToDraw =
            std::min(
                static_cast<size_t>(40),
                goodMatches.size());

        for (size_t i = 0;
            i < matchesToDraw;
            i++) {

            const DMatch& match =
                goodMatches[i];

            Point2f mapPoint =
                activeMapKeypoints[
                    match.queryIdx].pt;

            Point2f framePoint =
                frameKeypoints[
                    match.trainIdx].pt;

            framePoint.x +=
                static_cast<float>(
                    mapWidth);

            Scalar color =
                siftValid
                ? Scalar(0, 255, 0)
                : Scalar(0, 100, 255);

            line(
                result,
                mapPoint,
                framePoint,
                color,
                1,
                LINE_AA);

            circle(
                result,
                mapPoint,
                3,
                color,
                FILLED,
                LINE_AA);
        }

        // ====================================================
        // ОБЪЕДИНЁННАЯ ТРАЕКТОРИЯ
        //
        // Пока симуляции нет, полностью совпадает
        // с траекторией SIFT.
        // ====================================================
        for (size_t i = 1;
            i < trajectory.size();
            i++) {

            line(
                result,
                trajectory[i - 1],
                trajectory[i],
                Scalar(255, 255, 0),
                2,
                LINE_AA);
        }

        // ====================================================
        // ДОБАВЛЕНО:
        // ОТДЕЛЬНАЯ ТРАЕКТОРИЯ БИНС
        //
        // Она появится только после подключения симуляции.
        // ====================================================
        const vector<Point2f>& binsTrajectory =
            trajectoryCalculator.getBinsTrajectory();

        for (size_t i = 1;
            i < binsTrajectory.size();
            i++) {

            line(
                result,
                binsTrajectory[i - 1],
                binsTrajectory[i],
                Scalar(255, 0, 255),
                1,
                LINE_AA);
        }

        if (initialPositionFound) {
            circle(
                result,
                binsPosition,
                12,
                Scalar(255, 0, 255),
                2,
                LINE_AA);

            putText(
                result,
                "BINS",
                Point(
                    cvRound(
                        binsPosition.x - 18),
                    cvRound(
                        binsPosition.y - 16)),
                FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar(255, 0, 255),
                2,
                LINE_AA);
        }

        if (siftValid) {
            circle(
                result,
                siftPosition,
                6,
                Scalar(0, 255, 255),
                FILLED,
                LINE_AA);

            putText(
                result,
                "SIFT",
                Point(
                    cvRound(
                        siftPosition.x + 8),
                    cvRound(
                        siftPosition.y + 5)),
                FONT_HERSHEY_SIMPLEX,
                0.45,
                Scalar(0, 255, 255),
                1,
                LINE_AA);
        }

        // Информационная панель.
        rectangle(
            result,
            Point(0, contentHeight),
            Point(result.cols, result.rows),
            Scalar(35, 35, 35),
            FILLED);

        int textY =
            contentHeight + 25;

        char text[500];

        sprintf_s(
            text,
            sizeof(text),
            "Frame: %d | Mode: %s | SIFT: %s",
            frameID,
            mode.c_str(),
            siftValid
            ? "ACTIVE"
            : "LOST");

        putText(
            result,
            text,
            Point(10, textY),
            FONT_HERSHEY_SIMPLEX,
            0.52,
            siftValid
            ? Scalar(0, 255, 0)
            : Scalar(0, 100, 255),
            2,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "Reason: %s",
            failureReason.c_str());

        putText(
            result,
            text,
            Point(10, textY + 28),
            FONT_HERSHEY_SIMPLEX,
            0.47,
            Scalar(255, 255, 255),
            1,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "Frame points: %zu | Map points: %zu | Matches: %d | Inliers: %d (%.2f)",
            frameKeypoints.size(),
            activeMapKeypoints.size(),
            goodMatchesCount,
            inlierCount,
            inlierRatio);

        putText(
            result,
            text,
            Point(10, textY + 56),
            FONT_HERSHEY_SIMPLEX,
            0.45,
            Scalar(200, 200, 200),
            1,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "BINS radius: %.1f px | Lost: %d | Scale: %.2f | Motion: %s",
            binsErrorRadius,
            lostFrames,
            frameScale,
            simulationData.available
            ? "BINS+SIFT"
            : "SIFT ONLY");

        putText(
            result,
            text,
            Point(10, textY + 84),
            FONT_HERSHEY_SIMPLEX,
            0.45,
            Scalar(200, 200, 255),
            1,
            LINE_AA);

        if (initialPositionFound) {
            sprintf_s(
                text,
                sizeof(text),
                "Position: (%.1f, %.1f) px | Distance: %.1f px | Correction: %.1f px",
                fusedPosition.x,
                fusedPosition.y,
                trajectoryCalculator
                .getDistancePixels(),
                trajectoryCalculator
                .getLastCorrectionPixels());

            putText(
                result,
                text,
                Point(10, textY + 112),
                FONT_HERSHEY_SIMPLEX,
                0.47,
                Scalar(0, 255, 255),
                1,
                LINE_AA);
        }

        imshow(
            "Drone Navigation - Reliable BINS + SIFT",
            result);

        int key = waitKey(1);

        if (key == 27) {
            break;
        }

        if (key == 's' ||
            key == 'S') {

            string screenshotPath =
                "C:\\Diplom\\OpenCV_SIFT\\x64\\Debug\\screenshot_" +
                to_string(frameID) +
                ".png";

            imwrite(
                screenshotPath,
                result);
        }
    }

    double successPercent =
        processedFrames > 0.0
        ? successfulFrames /
        processedFrames *
        100.0
        : 0.0;

    cout << endl;
    cout << "========================================" << endl;
    cout << "FINAL STATISTICS" << endl;
    cout << "========================================" << endl;

    cout << "Total frames:    "
        << processedFrames << endl;

    cout << "Successful SIFT: "
        << successfulFrames << endl;

    cout << "Success:         "
        << fixed << setprecision(1)
        << successPercent << "%"
        << endl;

    // ДОБАВЛЕНО:
    cout << "Flight distance: "
        << fixed << setprecision(1)
        << trajectoryCalculator.getDistancePixels()
        << " px"
        << endl;

    cout << "Motion data:     "
        << (trajectoryCalculator.hasMotionData()
            ? "AVAILABLE"
            : "NOT AVAILABLE / SIFT ONLY")
        << endl;

    cout << "========================================" << endl;

    trajectoryFile.close();
    cap.release();
    destroyAllWindows();

    return 0;
}