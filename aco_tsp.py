import numpy as np
from scipy import spatial

# Данный код был написан с помощью статьи https://vc.ru/newtechaudit/353372-muravi-i-python-ishchem-samye-korotkie-puti
# и модернизирован

class ACO_TSP:  # класс алгоритма муравьиной колонии для решения задачи коммивояжёра
    def __init__(self, coordinates, number_of_ants=10, number_of_iter=20, alpha=1, beta=2, rho=0.1):
        self.n_dim = np.shape(coordinates)[0]  # количество точек
        self.number_of_ants = number_of_ants  # количество муравьёв
        self.number_of_iter = number_of_iter  # количество итераций
        self.alpha = alpha  # коэффициент важности феромонов в выборе пути
        self.beta = beta  # коэффициент значимости расстояния
        self.rho = rho  # скорость испарения феромонов
        self.distance_matrix = spatial.distance.cdist(coordinates, coordinates, metric='euclidean')
        self.prob_matrix_distance = 1 / (self.distance_matrix + 1e-10 * np.eye(self.n_dim, self.n_dim))

        # Матрица феромонов, обновляющаяся каждую итерацию
        self.Tau = np.ones((self.n_dim, self.n_dim))
        # Путь каждого муравья в определённом поколении
        self.Table = np.zeros((number_of_ants, self.n_dim)).astype(int)
        self.y = None  # Общее расстояние пути муравья в определённом поколении
        self.generation_best_X, self.generation_best_Y = [], [] # фиксирование лучших поколений
        self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y
        self.best_x, self.best_y = None, None

    def total_distance(self, routine):
        num_points, = routine.shape
        return sum([self.distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]]
                    for i in range(num_points)])
    def run(self, max_iter=None):
        self.number_of_iter = max_iter or self.number_of_iter
        for i in range(self.number_of_iter):
            # вероятность перехода без нормализации
            prob_matrix = (self.Tau ** self.alpha) * self.prob_matrix_distance ** self.beta
            for j in range(self.number_of_ants):  # для каждого муравья
                # точка начала пути (она может быть случайной, это не имеет значения)
                self.Table[j, 0] = 0
                for k in range(self.n_dim - 1):  # каждая вершина, которую проходят муравьи
                    # точка, которая была пройдена и не может быть пройдена повторно
                    taboo_set = set(self.Table[j, :k + 1])
                    # список разрешённых вершин, из которых будет происходить выбор
                    allow_list = list(set(range(self.n_dim)) - taboo_set)
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum() # нормализация вероятности
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point

            # рассчёт расстояния
            y = np.array([self.total_distance(i) for i in self.Table])

            # фиксация лучшего решения
            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.generation_best_X.append(x_best)
            self.generation_best_Y.append(y_best)

            # подсчёт феромона, который будет добавлен к ребру
            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.number_of_ants):  # для каждого муравья
                for k in range(self.n_dim - 1):  # для каждой вершины
                    # муравьи перебираются из вершины n1 в вершину n2
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]
                    delta_tau[n1, n2] += 1 / y[j]  # нанесение феромона
                # муравьи ползут от последней вершины обратно к первой
                n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]
                delta_tau[n1, n2] += 1 / y[j]  # нанесение феромона
            self.Tau = (1 - self.rho) * self.Tau + delta_tau
        best_generation = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[best_generation]
        self.best_y = self.generation_best_Y[best_generation]
        return self.best_x, self.best_y
    fit = run
