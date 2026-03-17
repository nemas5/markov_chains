from typing import List
import itertools as it
from random import choices
from random import randint, seed
from collections import deque

import numpy as np
from numpy.typing import NDArray as Arr
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def draw_labeled_multigraph(G, attr_name, ax=None):
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", connectionstyle=connectionstyle, ax=ax, width=2, node_size=1000
    )
    labels = {
        tuple(edge): f"{attr_name}={attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )


def draw_graph(matrix: List[List[float]], nodes: list) -> None:
    graph = nx.MultiDiGraph()
    ln = len(matrix)
    node_names = [''] * ln
    for i in range(ln):
        node_names[i] = "S" + str(i + 1) + '\n' + str(nodes[i])
        graph.add_node(node_names[i])
    for i in range(ln):
        for j in range(ln):
            if matrix[i][j] != 0:
                graph.add_edge(node_names[i], node_names[j], l=matrix[i][j])
    draw_labeled_multigraph(graph, "l")
    plt.show()


# Предполагается нагруженный резерв A и ненагруженный резерв B
# Минимум элементов для A = 1
def fill_matrix(na, nb, ra, rb, lma, lmb, lnu) -> tuple:
    s = dict()
    s_list = set()

    def rec_graph(a, b) -> tuple:
        if not((a, b) in s):
            if not((a, b) in s_list):
                s_list.add((a, b))
            if a > 0 and b > 0:
                s[(a, b)] = {"a": rec_graph(a - 1, b), "b": rec_graph(a, b - 1)}
            elif a == 0 and b > 0:
                s[(a, b)] = {"b": rec_graph(a, b - 1)}
            elif a > 0 and b == 0:
                s[(a, b)] = {"a": rec_graph(a - 1, b)}
            else:
                s[(a, b)] = dict()

            if 0 <= (a + b) < (na + nb + rb + ra):
                if (na + ra - a) > (nb + rb - b):
                    s[(a, b)]["nu"] = (a + 1, b)
                elif (na + ra - a) < (nb + rb - b):
                    s[(a, b)]["nu"] = (a, b + 1)
                elif lma >= lmb:
                    s[(a, b)]["nu"] = (a + 1, b)
                else:
                    s[(a, b)]["nu"] = (a, b + 1)
        return a, b

    ra = 0
    rec_graph(na + ra, nb + rb)
    ln = len(s_list)
    s_list = list(s_list)
    start = s_list.index((na + ra, nb + rb))
    s_list[start], s_list[0] = s_list[0], s_list[start]
    matrix = [[0. for j in range(ln)] for i in range(ln)]
    for i in range(ln):
        cur = s_list[i]
        if cur in s:
            if "a" in s[cur]:
                ja = s_list.index(s[cur]["a"])
                if cur[0] >= na:
                    matrix[i][ja] += lma * na
                else:
                    matrix[i][ja] += lma * cur[0]
            if "b" in s[cur]:
                jb = s_list.index(s[cur]["b"])
                if cur[1] < nb:
                    matrix[i][jb] += lmb * cur[1]
                else:
                    matrix[i][jb] += lmb * nb
            if "nu" in s[cur]:
                jnu = s_list.index(s[cur]["nu"])
                matrix[i][jnu] += lnu

    draw_graph(matrix, s_list)
    for i in range(ln):
        matrix[i][i] -= sum(matrix[i])
    print("\nМатрица Q:")
    for i in range(ln):
        print(matrix[i])
    print('\n')
    return matrix, s_list


def kolm_algebra(matrix: List[List[float]]):
    qt_matrix = np.transpose(matrix)
    ln = len(matrix)
    for i in range(ln):
        qt_matrix[-1][i] = 1
    right_part = [0. for i in range(ln)]
    right_part[-1] = 1
    pi_vector = np.linalg.solve(qt_matrix, right_part)
    print(f"\n Предельный вектор переходов:")
    print(pi_vector, '\n')
    return pi_vector


def math_exp(pi: Arr, order: list, mn_a: int, mn_b: int):
    f = 0
    a_ready = dict()
    b_ready = dict()
    for i in range(len(order)):
        a = order[i][0]
        b = order[i][1]
        if a < mn_a or b < mn_b:
            f += pi[i]
        if not(a in a_ready):
            a_ready[a] = 0
        a_ready[a] += pi[i]
        if not(b in b_ready):
            b_ready[b] = 0
        b_ready[b] += pi[i]
    me_a = 0
    me_b = 0
    for i, j in a_ready.items():
        me_a += i * j
    for i, j in b_ready.items():
        me_b += i * j
    print("Математическое ожидание количества готовых к эксплуатации устройств типа A: ", me_a)
    print("Математическое ожидание количества готовых к эксплуатации устройств типа B: ", me_b)
    print("Вероятность отказа системы: ", f)
    print("Коэффициент загрузки ремонтной службы: ", 1 - pi[0], '\n')


def solve(matrix, pi):
    ln = len(matrix)
    p0 = np.array([0. for i in range(ln)])
    p0[0] = 1.  # Начальные условия
    matrix = np.transpose(matrix)

    def sodu(t, p):
        p_dif = np.dot(matrix, p)
        return p_dif

    time_interval = (0, 0.36936936936936937 * 2)
    t_eval = np.linspace(*time_interval, 1000)
    sol = solve_ivp(sodu, time_interval, p0, t_eval=t_eval, method='RK45')

    # Визуализация
    for i in range(len(sol.y)):
        plt.plot(sol.t, sol.y[i], label=f"P({i})")
    plt.xlabel("t")
    plt.ylabel("P(k)")
    plt.legend()
    plt.grid()
    plt.show()

    for i in range(len(sol.t)):
        pi_cur = [sol.y[j][i] for j in range(len(sol.y))]
        if np.linalg.norm(pi - pi_cur) < np.linalg.norm(pi) * 0.01:
            t_end = sol.t[i]
            print("Теоретическая оценка времени переходного процесса", sol.t[i])
            return t_end


def imitation(matrix, t_end):
    t_end *= 2
    ln = len(matrix)
    s = [i for i in range(ln)]
    t = 0.
    cur_s = 0
    res = dict()
    res[0] = 0
    while t < t_end:
        t_stay = np.random.exponential((-1) / matrix[cur_s][cur_s])
        p = [matrix[cur_s][i] / (- matrix[cur_s][cur_s]) if i != cur_s else 0 for i in range(ln)]
        cur_s = choices(s, p)[0]
        t += t_stay
        res[t] = cur_s
    print("\nМоделирование в терминах непрерывных цепей Маркова:")
    print(res)


class WorkingDevice:
    def __init__(self, start, num, intensity):
        self.__id = num
        self.__intense = intensity
        self.__break_down = self.new_break_down_time(start)

    def new_break_down_time(self, start):
        return start + np.random.exponential(1 / self.__intense)

    def upd_break_down_time(self, start):
        self.__break_down = self.new_break_down_time(start)

    def get_break_down(self):
        return self.__break_down

    def get_id(self):
        return self.__id


def discrete_modeling(na, nb, ra, rb, lma, lmb, lnu):
    ra = 0
    log_file = open("log.txt", "w", encoding="utf-8")

    type_a_active = {WorkingDevice(0, i, lma) for i in range(na)}
    type_b_active = {WorkingDevice(0, i, lmb) for i in range(nb)}
    type_b_reserve = {WorkingDevice(0, i + nb, lmb) for i in range(rb)}
    repair_que_a = deque()
    repair_que_b = deque()

    current_time = 0.
    end_time = 0.3689996302608915 * 2
    repair_start = current_time
    while current_time < end_time:
        next_repair = end_time * 2
        next_repair_type = "a"
        next_break_a = WorkingDevice(end_time * 2, 1000, lma)
        next_break_b = WorkingDevice(end_time * 2, 1000, lmb)
        print(f"Время до - {current_time}:", file=log_file)

        # Выясняем приоритет на починку
        if len(repair_que_a) > len(repair_que_b):
            next_repair = repair_start + repair_que_a[0][1]  # (WorkingDevice, repairing_time)
            next_repair_type = "a"
            print("Для починки приоритетно устройство типа A (количество)", file=log_file)
        elif len(repair_que_b) > len(repair_que_a):
            next_repair = repair_start + repair_que_b[0][1]
            next_repair_type = "b"
            print("Для починки приоритетно устройство типа B (количество)", file=log_file)
        elif len(repair_que_b) + len(repair_que_a) == 0:
            print("Устройства не нуждаются в ремонте", file=log_file)
        elif lma > lmb:
            next_repair = repair_start + repair_que_a[0][1]
            next_repair_type = "a"
            print("Для починки приоритетно устройство типа A (интенсивность)", file=log_file)
        else:
            next_repair = repair_start + repair_que_b[0][1]
            next_repair_type = "b"
            print("Для починки приоритетно устройство типа B (интенсивность)", file=log_file)

        # Какое следующее устройство типа A сломается
        if len(type_a_active):
            next_break_a = min(type_a_active, key=lambda dev: dev.get_break_down())

        # Какое следующее устройство типа B сломается
        if len(type_b_active):
            next_break_b = min(type_b_active, key=lambda dev: dev.get_break_down())

        # Следующее событие - починка
        print(next_repair, next_break_b.get_break_down(), next_break_a.get_break_down(), file=log_file)
        if next_repair <= next_break_b.get_break_down() and next_repair <= next_break_a.get_break_down():
            # Чиним A
            if next_repair_type == "a":
                new_event = repair_que_a.popleft()
                new_event[0].upd_break_down_time(current_time + new_event[1])
                type_a_active.add(new_event[0])
                print(f"Починено устройство типа A, номер {new_event[0].get_id()},"
                      f" введено в эксплуатацию", file=log_file)
            # Чиним B
            else:
                new_event = repair_que_b.popleft()
                new_event[0].upd_break_down_time(current_time + new_event[1])  # проверить, где меняется
                if len(type_b_active) == nb:
                    type_b_reserve.add(new_event[0])
                    print(f"Починено устройство типа B, номер {new_event[0].get_id()},"
                          f" пополнило резерв", file=log_file)
                else:
                    type_b_active.add(new_event[0])
                    print(f"Починено устройство типа B, номер {new_event[0].get_id()},"
                          f" введено в эксплуатацию", file=log_file)
            current_time += new_event[1]
            repair_start = current_time
            # new_event[0].new_break_down_time()  # Мб можно так?

        # Следующее событие - поломка устройства типа B
        elif next_break_b.get_break_down() <= next_repair \
                and next_break_b.get_break_down() <= next_break_a.get_break_down():
            current_time = next_break_b.get_break_down()
            type_b_active.remove(next_break_b)
            if len(repair_que_a) + len(repair_que_b) == 0:
                repair_start = current_time
            repair_que_b.append((next_break_b, np.random.exponential(1 / lnu)))  # ТОЛЬКО время починки
            print(f"Сломалось устройство типа B, номер {next_break_b.get_id()}",
                  file=log_file)
            if len(type_b_reserve) > 0:  # Переход из резерва
                new_reserve = type_b_reserve.pop()
                type_b_active.add(new_reserve)
                print(f"Задействован резервный элемент B, номер {new_reserve.get_id()}",
                      file=log_file)

        # Следующее событие - поломка устройства типа A
        else:
            current_time = next_break_a.get_break_down()
            if len(repair_que_a) + len(repair_que_b) == 0:
                repair_start = current_time
            type_a_active.remove(next_break_a)
            repair_que_a.append((next_break_a, np.random.exponential(1 / lnu)))
            print(f"Сломалось устройство типа A, номер {next_break_a.get_id()}",
                  file=log_file)

        # Работоспособность системы
        print(f"Время после - {current_time}:", file=log_file)
        if len(type_a_active) > 0 and len(type_b_active) >= nb:
            print(f"Система в рабочем состоянии", file=log_file)
        else:
            print("Система стоит", file=log_file)
        print("\n", file=log_file)

    log_file.close()


if __name__ == '__main__':
    N = 175
    G = 6

    lam_a = G + (N % 3)
    lam_b = G + (N % 5)
    N_A = 2 + (G % 2)
    N_B = 1 + (N % 2)
    R_A = 1 + (G % 2)
    R_B = 2 - (G % 2)
    NU = (N_A + N_B - (G % 2)) * (G + (N % 4))

    print(f"lam_a = {lam_a}")
    print(f"lam_b = {lam_b}")
    print(f"N_A = {N_A}")
    print(f"N_B = {N_B}")
    print(f"R_A = {R_A}")
    print(f"R_B = {R_B}")
    print(f"NU = {NU}")

    Q, directions = fill_matrix(N_A, N_B, R_A, R_B, lam_a, lam_b, NU)
    PI = kolm_algebra(Q)
    math_exp(PI, directions, 1, N_B)
    T = solve(Q, PI)
    imitation(Q, T)

    print("\nДискретно-событийное моделирование:\n")
    discrete_modeling(N_A, N_B, R_A, R_B, lam_a, lam_b, NU)
