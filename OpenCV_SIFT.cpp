#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <windows.h>

using namespace cv;
using namespace std;

int main() {
    // Поддержка русского языка в консоли
    setlocale(LC_ALL, "Russian");

    cout << "Запуск программы..." << endl;

    // Загрузка изображений (АБСОЛЮТНЫЙ ПУТЬ)
    Mat map = imread("C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\map.jpg", IMREAD_GRAYSCALE);
    Mat frame = imread("C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\frame.jpg", IMREAD_GRAYSCALE);

    // Проверка загрузки
    if (map.empty() || frame.empty()) {
        cout << "Ошибка загрузки изображений!" << endl;
        return -1;
    }

    cout << "Изображения успешно загружены!" << endl;

    // --- SIFT детектор ---

    // ❌ СТАРЫЙ ВАРИАНТ (без ограничения количества точек)
    // Ptr<SIFT> sift = SIFT::create();

    // ✅ НОВЫЙ ВАРИАНТ (ограничиваем количество точек)
    Ptr<SIFT> sift = SIFT::create(150); // 100–200 — оптимально

    vector<KeyPoint> kp1, kp2;
    Mat des1, des2;

    sift->detectAndCompute(map, noArray(), kp1, des1);
    sift->detectAndCompute(frame, noArray(), kp2, des2);

    cout << "Ключевых точек (map): " << kp1.size() << endl;
    cout << "Ключевых точек (frame): " << kp2.size() << endl;

    // --- Сопоставление дескрипторов ---
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn_matches;

    matcher.knnMatch(des1, des2, knn_matches, 2);

    // Фильтр Лоу
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() < 2) continue;

        if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    cout << "Хорошие совпадения: " << good_matches.size() << endl;

    // --- Отрисовка совпадений ---
    Mat result;
    drawMatches(map, kp1, frame, kp2, good_matches, result);

    // --- Показ окна ---
    imshow("Совпадения SIFT", result);

    cout << "Нажмите любую клавишу..." << endl;
    waitKey(0);

    return 0;
}