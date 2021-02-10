import numpy as np
import time

eps = 0.0001


def head_of_table():
    print('{0:>37}  | {1}'.format('Норма', 'Оценка'))
    print('{0:>9}|{1:>9}|{2:>9}|{3:>9}|{4:>12}|{5:>9} {6:>9} {7:>9} {8:>9}'.format('Итерация', 'tau', 'q',
                                                                                   'невязки', 'погрешности',
                                                                                   'x[1]', 'x[2]', 'x[3]',
                                                                                   'x[4]'))


def data_print(i, tau, q, norma, err, x):
    print('{0:>9}|{1:>9.5f}|{2:>9.5f}|{3:>9.5f}|{4:>12.5f}|{5:>9.5f} {6:>9.5f} {7:>9.5f} {8:>9.5f}'.format(i, tau, q,
                                                                                                           norma, err,
                                                                                                           x[0], x[1],
                                                                                                           x[2], x[3]))


def simple_it(A, b, x):
    print('\nМетод простых итераций')
    head_of_table()
    iteration = 0
    tau = 1.8 / np.linalg.norm(A, 1)
    check = True
    x_prev = x = np.copy(b)
    while check:
        iteration += 1
        x_next = np.array([0., ] * len(A))
        for i in range(len(A)):
            sum = 0.0
            for j in range(len(A)):
                sum += A[i][j] * x[j]
            x_next[i] = x[i] + tau * (b[i] - sum)
        q = np.linalg.norm(x_next - x, np.inf) / np.linalg.norm(x, np.inf) if iteration == 1 else np.linalg.norm(
            x_next - x, np.inf) / np.linalg.norm(x - x_prev, np.inf)
        norm_of_disc = np.linalg.norm(A.dot(x_next) - b, np.inf)
        err = np.linalg.norm(x_next - x, np.inf) * q / (1 - q)
        x_prev = np.copy(x)
        x = np.copy(x_next)
        data_print(iteration, tau, q, norm_of_disc, err, x)
        if norm_of_disc < eps:
            check = False


def fast_descent_method(A, b, x):
    print('\nМетод скорейшего спуска')
    head_of_table()
    x_prev = x = np.copy(b)
    iteration = 0
    check = True
    while check:
        iteration += 1
        tmp = A.dot(x) - b
        tau = np.dot(tmp, tmp) / np.dot(A.dot(tmp), tmp)
        x_next = np.array([0., ] * len(A))
        for i in range(len(A)):
            sum = 0.0
            for j in range(len(A)):
                sum += A[i][j] * x[j]
            x_next[i] = x[i] + tau * (b[i] - sum)
        q = np.linalg.norm(x_next - x, np.inf) / np.linalg.norm(x, np.inf) if iteration == 1 else np.linalg.norm(
            x_next - x, np.inf) / np.linalg.norm(x - x_prev, np.inf)
        norm_of_disc = np.linalg.norm(A.dot(x_next) - b, np.inf)
        err = np.linalg.norm(x_next - x, np.inf) * q / (1 - q)
        x_prev = np.copy(x)
        x = np.copy(x_next)
        data_print(iteration, tau, q, norm_of_disc, err, x)
        if norm_of_disc < eps:
            check = False


def SOR(A, b, x):
    print('\nМетод ПВР')
    iteration = iteration_min = 0
    omega = omega_optimal = 0.1
    while omega < 2:
        iteration1 = 0
        x = np.copy(b)
        x_next = np.array([0., ] * len(A))
        check = True
        while check:
            iteration1 += 1
            for i in range(len(A)):
                sum = 0.0
                for j in range(len(A)):
                    if i != j:
                        sum += A[i][j] * x_next[j]
                x_next[i] = x[i] + omega * ((1 / A[i][i]) * (b[i] - sum) - x[i])
            norm_of_disc = np.linalg.norm(A.dot(x_next) - b, np.inf)
            x = np.copy(x_next)
            if norm_of_disc < eps:
                check = False
        if omega == 0.1 or iteration1 < iteration_min:
            iteration_min = iteration1
            omega_optimal = omega
        omega += 0.1
    print('Оптимальный параметр релаксации =', omega_optimal.__format__('.1f'), 'при', iteration_min, 'итерациях')
    check = True
    x_prev = x = np.copy(b)
    x_next = np.array([0., ] * len(A))
    head_of_table()
    while check:
        iteration += 1
        for i in range(len(A)):
            sum = 0.0
            for j in range(len(A)):
                if i != j:
                    sum += A[i][j] * x_next[j]
            x_next[i] = x[i] + omega_optimal * ((1 / A[i][i]) * (b[i] - sum) - x[i])
        q = np.linalg.norm(x_next - x, np.inf) / np.linalg.norm(x, np.inf) if iteration == 1 else np.linalg.norm(
            x_next - x, np.inf) / np.linalg.norm(x - x_prev, np.inf)
        norm_of_disc = np.linalg.norm(A.dot(x_next) - b, np.inf)
        err = np.linalg.norm(x_next - x, np.inf) * q / (1 - q)
        x_prev = np.copy(x)
        x = np.copy(x_next)
        data_print(iteration, omega_optimal, q, norm_of_disc, err, x)
        if norm_of_disc < eps:
            check = False


def conjugate_coefficient_method(A, b, x):
    print('\nМетод сопряженных коеффициентов')
    head_of_table()
    iteration = 0
    tau_prev = 0
    check = True
    x_prev = x = np.copy(b)
    alpha = 1
    while check:
        iteration += 1
        x_next = np.array([0., ] * len(A))
        tmp = A.dot(x) - b
        tau = np.dot(tmp, tmp) / np.dot(A.dot(tmp), tmp)
        alpha = 1 if iteration == 1 else pow(
            1 - tau / tau_prev / alpha * np.dot(A.dot(x) - b, A.dot(x) - b) / np.dot(A.dot(x_prev) - b,
                                                                                     A.dot(x_prev) - b), -1)
        for i in range(len(A)):
            x_next[i] = alpha*x[i] + (1-alpha) * x_prev[i]-tau*alpha*(A.dot(x) - b)[i]
        q = np.linalg.norm(x_next - x, np.inf) / np.linalg.norm(x, np.inf) if iteration == 1 else np.linalg.norm(
            x_next - x, np.inf) / np.linalg.norm(x - x_prev, np.inf)
        norm_of_disc = np.linalg.norm(A.dot(x_next) - b, np.inf)
        err = np.linalg.norm(x_next - x, np.inf) * q / (1 - q)
        x_prev = np.copy(x)
        x = np.copy(x_next)
        tau_prev=tau
        data_print(iteration, tau, q, norm_of_disc, err, x)
        if norm_of_disc < eps:
            check = False


file = open('in.txt', 'r')
A = np.array([[float(el) for el in line.split()] for line in file], float)
print("А:\n", A)
file.close()
b = np.array([(float)(i + 1) for i in range(len(A))])
print("b:\n", b)
x = np.array([0., ] * len(A))
time_start = time.time()
simple_it(A, b, x)
time_finish = time.time()
time_simple_it = time_finish - time_start
time_start = time.time()
fast_descent_method(A, b, x)
time_finish = time.time()
time_fast_descent = time_finish - time_start
time_start = time.time()
SOR(A, b, x)
time_finish = time.time()
time_SOR = time_finish - time_start
time_start = time.time()
conjugate_coefficient_method(A, b, x)
time_finish = time.time()
time_conjugate_coefficient_method = time_finish - time_start
print('\nВремя выполнения метода простой итерации = {0:.1f} мс'.format(time_simple_it*1000))
print('Время выполнения метода скорейшего спуска = {0:.1f} мс'.format(time_fast_descent*1000))
print('Время выполнения метода ПВР = {0:.1f} мс'.format(time_SOR*1000))
print('Время выполнения метода сопряженных коэффициентов = {0:.1f} мс'.format(time_conjugate_coefficient_method*1000))
norms = np.linalg.norm(A, 1) * np.linalg.norm(np.linalg.inv(A), 1)
print('\nCond(A) = ', norms.__format__('0.5f'))
print('\nТеоретическая оценка числа итераций:')
print('Метод простых итераций: ', (np.log(1/eps)/2*norms).__format__('0.0f'))
print('Метод скорейшего спуска: ', (np.log(1 / eps) / 2 * norms).__format__('0.0f'))
print('Метод ПВР: ', (np.log(1 / eps) / 4 * np.sqrt(norms)).__format__('0.0f'))
print('Метод сопряженных градиентов: ', (np.log(2 / eps) / 2 * np.sqrt(norms)).__format__('0.0f'))

del (A, b, x)
