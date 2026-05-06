#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <windows.h>

using namespace cv;
using namespace std;

int main() {
    // Установка русской локали для корректного вывода сообщений в консоль
    setlocale(LC_ALL, "Russian");

    cout << "Запуск программы..." << endl;

    // ===== ЗАГРУЗКА ЭТАЛОННОЙ КАРТЫ =====
    // Загружаем изображение карты/ориентира в градациях серого
    Mat map = imread("C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\map.jpg", IMREAD_GRAYSCALE);

    // Проверка: удалось ли загрузить изображение
    if (map.empty()) {
        cout << "Ошибка загрузки карты!" << endl;
        return -1;
    }

    // Выравнивание гистограммы - улучшает контрастность изображения
    equalizeHist(map, map);
    // Применение размытия по Гауссу для уменьшения шума
    GaussianBlur(map, map, Size(3, 3), 0);

    // ===== ЗАГРУЗКА ВИДЕО =====
    // Открываем видеофайл для обработки
    VideoCapture cap("C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\video.mp4");

    // Проверка: удалось ли открыть видео
    if (!cap.isOpened()) {
        cout << "Ошибка открытия видео!" << endl;
        return -1;
    }

    cout << "Видео запущено!" << endl;

    // ===== НАСТРОЙКА SIFT (Scale-Invariant Feature Transform) =====
    // Создаём детектор SIFT с максимум 1000 ключевых точек на изображение
    Ptr<SIFT> sift = SIFT::create(1000);

    // Детектируем ключевые точки и вычисляем дескрипторы для эталонной карты
    vector<KeyPoint> kp_map;    // Ключевые точки карты
    Mat des_map;                // Дескрипторы карты (числовые характеристики точек)
    sift->detectAndCompute(map, noArray(), kp_map, des_map);

    cout << "Ключевых точек (map): " << kp_map.size() << endl;

    // Создаём BFMatcher (Brute-Force Matcher) с евклидовым расстоянием (L2)
    BFMatcher matcher(NORM_L2);

    // ===== ФИЛЬТР КАЛМАНА =====
    // 4 состояния: [x, y, vx, vy] (положение и скорость по осям)
    // 2 измерения: [x, y] (наблюдаемые координаты)
    KalmanFilter kf(4, 2, 0);

    float dt = 1.0f;  // Временной шаг между кадрами

    // Матрица перехода состояния: предсказывает новое положение на основе предыдущего
    // x_new = x + vx*dt
    // y_new = y + vy*dt
    // vx_new = vx
    // vy_new = vy
    kf.transitionMatrix = (Mat_<float>(4, 4) <<
        1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1);

    // Матрица измерений: из состояния [x,y,vx,vy] извлекаем только [x,y]
    kf.measurementMatrix = (Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    // Ковариация шума процесса (доверие к модели движения)
    setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
    // Ковариация шума измерений (доверие к наблюдениям)
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
    // Начальная ковариация ошибки
    setIdentity(kf.errorCovPost, Scalar::all(1));

    // Начальное состояние: все нули
    kf.statePost = (Mat_<float>(4, 1) << 0, 0, 0, 0);

    Mat measurement = Mat::zeros(2, 1, CV_32F);  // Вектор для измерений
    bool kalman_initialized = false;             // Флаг инициализации фильтра

    int frame_id = 0;          // Счётчик кадров
    Point2f last_position;     // Последняя достоверная позиция
    bool has_position = false;  // Есть ли текущая позиция

    // ===== ГЛАВНЫЙ ЦИКЛ ОБРАБОТКИ ВИДЕО =====
    while (true) {

        Mat frame, gray;
        cap >> frame;  // Захватываем следующий кадр

        if (frame.empty()) break;  // Если кадров больше нет - выходим

        frame_id++;

        // Обрабатываем каждый 5-й кадр для повышения производительности
        if (frame_id % 5 != 0) continue;

        // Преобразуем кадр в оттенки серого
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        // Улучшаем контрастность
        equalizeHist(gray, gray);
        // Уменьшаем шум
        GaussianBlur(gray, gray, Size(3, 3), 0);

        // Детектируем ключевые точки и дескрипторы для текущего кадра
        vector<KeyPoint> kp_frame;
        Mat des_frame;
        sift->detectAndCompute(gray, noArray(), kp_frame, des_frame);

        // Если дескрипторов нет - пропускаем кадр
        if (des_frame.empty()) continue;

        // ===== ПОИСК СООТВЕТСТВИЙ МЕЖДУ ЭТАЛОНОМ И КАДРОМ =====
        vector<vector<DMatch>> knn_matches;
        // Находим 2 ближайших соседа для каждого дескриптора эталона
        matcher.knnMatch(des_map, des_frame, knn_matches, 2);

        vector<DMatch> good_matches;  // Хорошие совпадения

        // Применяем тест Лоу (Lowe's ratio test) для фильтрации ложных совпадений
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() < 2) continue;

            // Сохраняем совпадение, если лучшее значительно лучше второго (коэф. 0.75)
            if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        cout << "Кадр: " << frame_id
            << " | совпадений: " << good_matches.size();

        Point2f current_position;
        bool valid_position = false;
        int inliers_count = 0;

        Mat mask;  // Маска для хранения inliers после RANSAC

        // ===== ВЫЧИСЛЕНИЕ ГОМОГРАФИИ =====
        // Нужно минимум 10 совпадений для надёжного расчёта
        if (good_matches.size() >= 10) {

            vector<Point2f> pts_map, pts_frame;

            // Собираем координаты соответствующих точек
            for (auto& m : good_matches) {
                pts_map.push_back(kp_map[m.queryIdx].pt);    // Точка на эталоне
                pts_frame.push_back(kp_frame[m.trainIdx].pt); // Точка на кадре
            }

            // Находим гомографию (матрицу преобразования) методом RANSAC
            // Вход: точки кадра -> выход: соответствующие точки карты
            Mat H = findHomography(pts_frame, pts_map, RANSAC, 5.0, mask);

            if (!H.empty()) {

                // Количество inliers - точек, согласующихся с найденной гомографией
                inliers_count = countNonZero(mask);

                // Берём центр текущего кадра
                vector<Point2f> frame_center(1, Point2f(gray.cols / 2, gray.rows / 2));
                vector<Point2f> map_center;

                // Преобразуем центр кадра в координаты на эталонной карте
                perspectiveTransform(frame_center, map_center, H);

                current_position = map_center[0];  // Текущая позиция на карте

                // Считаем позицию достоверной, если много inliers
                if (inliers_count >= 8)
                    valid_position = true;

                cout << " | inliers: " << inliers_count;
            }
        }

        // ===== ПРИМЕНЕНИЕ ФИЛЬТРА КАЛМАНА =====
        // Предсказываем следующее состояние на основе текущего
        Mat prediction = kf.predict();
        Point2f predicted(prediction.at<float>(0), prediction.at<float>(1));

        bool used_prediction = false;

        // ОТБРОС ВЫБРОСОВ: если позиция слишком резко изменилась (>120 пикселей)
        if (valid_position && has_position) {
            if (norm(current_position - last_position) > 120) {
                valid_position = false;  // Отклоняем как выброс
                cout << " | ОТКЛОНЕНО";
            }
        }

        if (valid_position) {
            // Если это первое достоверное измерение - инициализируем фильтр
            if (!kalman_initialized) {
                kf.statePost.at<float>(0) = current_position.x;
                kf.statePost.at<float>(1) = current_position.y;
                kf.statePost.at<float>(2) = 0;  // Начальная скорость по X
                kf.statePost.at<float>(3) = 0;  // Начальная скорость по Y
                kalman_initialized = true;
            }

            // Передаём измерение в фильтр Калмана
            measurement.at<float>(0) = current_position.x;
            measurement.at<float>(1) = current_position.y;
            kf.correct(measurement);  // Коррекция состояния

            last_position = current_position;
            has_position = true;

            cout << " | POS: (" << current_position.x << ", " << current_position.y << ")";
        }
        else if (has_position) {
            // Нет достоверного измерения - используем предсказание
            last_position = predicted;
            used_prediction = true;

            cout << " | PRED: (" << predicted.x << ", " << predicted.y << ")";
        }
        else {
            cout << " | нет позиции";
        }

        cout << endl;

        // ===== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ =====

        // ВИЗУАЛИЗАЦИЯ НА КАРТЕ
        Mat map_vis;
        cvtColor(map, map_vis, COLOR_GRAY2BGR);  // Преобразуем в цветную для отображения

        if (has_position) {
            if (used_prediction)
                // Синий кружок - предсказанная позиция
                circle(map_vis, last_position, 8, Scalar(255, 0, 0), -1);
            else
                // Зелёный кружок - измеренная позиция
                circle(map_vis, last_position, 8, Scalar(0, 255, 0), -1);
        }

        imshow("MAP", map_vis);       // Показываем карту с позицией

        // ВИЗУАЛИЗАЦИЯ КАДРА
        imshow("FRAME", frame);       // Показываем текущий кадр

        // ВИЗУАЛИЗАЦИЯ СОВПАДЕНИЙ
        Mat matches_img;

        if (!good_matches.empty()) {

            // Преобразуем маску inliers в вектор для отрисовки
            vector<char> mask_vec;
            if (!mask.empty()) {
                mask_vec.assign(mask.begin<uchar>(), mask.end<uchar>());
            }

            // Рисуем линии между соответствующими точками на карте и кадре
            drawMatches(
                map, kp_map,
                gray, kp_frame,
                good_matches,
                matches_img,
                Scalar::all(-1),   // Цвет ключевых точек (по умолчанию)
                Scalar::all(-1),   // Цвет совпадений (по умолчанию)
                mask_vec,          // Маска: отрисовываем только inliers
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  // Не рисуем одиночные точки
            );

            imshow("MATCHES", matches_img);  // Показываем совпадения
        }

        // Выход по клавише ESC (код 27)
        if (waitKey(1) == 27) break;
    }

    // Освобождаем ресурсы
    cap.release();
    destroyAllWindows();

    return 0;
}