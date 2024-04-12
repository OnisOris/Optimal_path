from aco_tsp import ACO_TSP
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


def rand_points_in_circle(a, b, r, num=10):
    """
  Функция генерирует num рандомных точек внутри окружности с центром (a, b) и радиусом r
  a - точка нуля по оси x
  b - точка нуля по оси y
  r - радиус окружности
  """
    start_x = a - r
    start_y = b - r
    points = np.array([0, 0])
    for i in range(num):
        rand_x = start_x + np.random.rand() * 2 * r
        if i % 2 == 0:
            size_y = -np.sqrt(r ** 2 - (rand_x - a) ** 2)
        else:
            size_y = np.sqrt(r ** 2 - (rand_x - a) ** 2)
        rand_y = b + size_y * np.random.rand()
        points = np.vstack([points, [rand_x, rand_y]])
    points = points[1:]

    dist = []
    for idx, point in enumerate(points):
        dist.append([distance(point, [a, b]), idx, points[idx, 0], points[idx, 1]])
    dist.sort(reverse=True)
    dist = np.array(dist)
    points_r = dist[:, 2:4]
    return points_r


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


dist_count = False  # если хотим посмотреть график оптимальных дистанций на разных итерациях, ставим True
a = 0  # координата x центра окружности
b = 0  # координата y центра окружности
r = int(sys.argv[2])  # радиус окружности
num = int(sys.argv[1])  # количество генерируемых точек
points_coordinate = rand_points_in_circle(a, b, r, num)  # генерация точек внутри круга
# Настройки для муравьиного алгоритма:
size_pop = 40  # количество муравьёв
max_iter = 200  # количество итераций
alpha = 1  # константы (подбирается вручную)
beta = 2  # константа (подбирается вручную)


def main():
    # создание объекта алгоритма муравьиной колонии
    aca = ACO_TSP(points_coordinate,
                  number_of_ants=size_pop,
                  number_of_iter=max_iter,
                  alpha=alpha,
                  beta=beta)
    best_x, best_y = aca.run()
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]

    # Отрисовка
    if dist_count:
        fig, ax = plt.subplots(1, 2)
        for index in range(0, len(best_points_)):  # аннотация индексов точек
            ax[0].annotate(best_points_[index], (best_points_coordinate[index, 0], best_points_coordinate[index, 1]))
        ax[0].plot(best_points_coordinate[:, 0],  # построение траекторий
                   best_points_coordinate[:, 1], 'o-g')
        pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    else:
        fig, ax = plt.subplots(1, 1)
        for index in range(0, len(best_points_)):  # аннотация индексов точек
            ax.annotate(best_points_[index], (best_points_coordinate[index, 0], best_points_coordinate[index, 1]))
        ax.plot(best_points_coordinate[:, 0],  # построение траекторий
                best_points_coordinate[:, 1], 'o-g')
        c = plt.Circle((a, b), radius=r, color='red', alpha=.3)
        plt.gca().add_artist(c)
        plt.xlim([a - r, a + r])
        plt.ylim([b - r, b + r])
        ax.scatter(points_coordinate[0, 0], points_coordinate[0, 1], edgecolors='#bcbd22', s=300)
        ax.scatter(a, b, edgecolors='#d62728', s=70)
        plt.legend(("Траектория обхода точек", f"Круг с радиусом r = {r}, и нулем в [{a}, {b}]",
                    "Сама дальняя точка от центра", "Центр"))
        plt.title(f"График траектории при популяции муравьев в {size_pop} особей")

    # изменение размера графиков
    plt.rcParams['figure.figsize'] = [20, 10]

    plt.show()


if __name__ == "__main__":
    main()
