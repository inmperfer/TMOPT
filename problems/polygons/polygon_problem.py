import multiprocessing
import random
import time
import traceback
import pickle as pk
from datetime import datetime

from optimization.neighborhood import VNS, VND
from optimization.problem import Problem
from optimization.tabu import TabuSearch
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import os

class PoligonProblem(Problem):
    def __init__(self,
                 target_image,
                 num_shapes,
                 candidates_by_iteration=100,
                 max_edges = 10,
                 delta=1,
                 neighborhood = 'all',
                 polygon_list=None,
                 sol_file='sol.png',
                 vns_vnd='None'):

        self.target_image = target_image
        self.target_image_diff = target_image.astype('int16')
        self.num_shapes = num_shapes
        self.candidates_by_iteration = candidates_by_iteration
        self.initilized_graph = False
        self.draw_queue = multiprocessing.Queue()
        self.draw_process = None
        self.polygon_list = polygon_list
        self.fitness_count = 0
        self.fitness_time = 0
        self.max_edges = max_edges
        self.delta = delta
        self.neighborhood = neighborhood
        self.h, self.w = target_image.shape[:2]

        # Modificamos el nombre del fichero de salida para que no se sobrescriba con las diferentes pruebas
        path, ext = os.path.splitext(sol_file)[:]
        self.sol_file = path + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + ext

        self.vns_vnd = vns_vnd


    def get_neighborhood(self, cand, neighborhood=None, num_candidates=None):
        if neighborhood is None:
            neighborhood = self.neighborhood

        if num_candidates is None:
            num_candidates = self.candidates_by_iteration

        candidates = []
        while len(candidates) < num_candidates:
            if neighborhood == 'all':
                _neighborhood = random.choice(['move', 'color', 'add', 'remove'])
                candidates += self.get_neighborhood(cand, _neighborhood, 1)
            elif neighborhood == 'move':
                candidates.append(self.__move__neighbor(cand))
            elif neighborhood == 'color':
                candidates.append(self.__color_neighbor(cand))
            elif neighborhood == 'add':
                poligons = cand
                if self.polygon_list:
                    poligons = [cand[i] for i in self.polygon_list]
                if min([len(p) for p, c in poligons]) < self.max_edges:
                    new_cand = self.__add__neighbor(cand)
                    if not new_cand is None:
                        candidates.append(new_cand)
                else:
                    candidates.append(cand)
            elif neighborhood == 'remove':
                poligons = cand
                if self.polygon_list:
                    poligons = [cand[i] for i in self.polygon_list]
                if max([len(p) for p, c in poligons]) > 3:
                    new_cand = self.__remove__neighbor(cand)
                    if not new_cand is None:
                        candidates.append(new_cand)
                else:
                    candidates.append(cand)
        return candidates


    # cand = [(((0, 0), (50, 0), (50, 20)), cand[0][1])]
    # Generacion de un vecino por eliminacion de un vértice y devolver candidato
    def __remove__neighbor(self, cand):
        if self.polygon_list:
            i = random.choice(self.polygon_list)
        else:
            i = random.randint(0, len(cand) - 1)
        cand = copy.deepcopy(cand)

        cand[i] = self.__remove_vertex(cand[i][0]), cand[i][1]
        return cand




    # Generacion de un vecino añadiendo un vértice y devolver candidato
    def __add__neighbor(self, cand):
        if self.polygon_list:
            i = random.choice(self.polygon_list)
        else:
            i = random.randint(0, len(cand) - 1)
        cand = copy.deepcopy(cand)

        cand[i] = self.__add_vertex(cand[i][0]), cand[i][1]
        return cand




    # Generacion de un vecino cambiando el color y devolver candidato
    def __color_neighbor(self, cand):
        if self.polygon_list:
            i = random.choice(self.polygon_list)
        else:
            i = random.randint(0, len(cand) - 1)
        cand = copy.deepcopy(cand)

        cand[i] = cand[i][0], self.__perturb_color(cand[i][1])
        return cand


    def __move__neighbor(self, cand):
        if self.polygon_list:
            i = random.choice(self.polygon_list)
        else:
            i = random.randint(0, len(cand) - 1)

        cand = copy.deepcopy(cand)

        j = random.randint(0, len(cand[i][0])-1)
        cand[i][0][j] = self.__move_point(*cand[i][0][j])
        return cand


    def __move_vertex(self, polygon):
        polygon = copy.deepcopy(polygon)
        i=random.randint(0, len(polygon) - 1)
        polygon[i] = self.__move_point(polygon[i][0], polygon[i][1])
        return polygon


    def __move_point(self, x, y):
        offset = 10
        if random.choice([True, False]):
            x = max(min(x + random.randint(-offset, offset), self.w), 0)
        else:
            y = max(min(y + random.randint(-offset, offset), self.h), 0)

        return x, y


    # Añade un vértice aleatorio al polígono que recibe como entrada
    def __add_vertex(self, polygon):
        if(len(polygon) < self.max_edges):
            polygon = copy.deepcopy(polygon)
            polygon.insert(random.randint(0, len(polygon) - 1), self.get_random_point())
        return polygon


    # Borra un vértice aleatorio al polígono que recibe como entrada
    def __remove_vertex(self, polygon):
        polygon = copy.deepcopy(polygon)
        polygon.remove(random.choice(polygon))
        return polygon

    # Modifica aleatoriamente el color que recibe como entrada
    def __perturb_color(self, color):
        offset = 10
        i = random.randint(0, 3)
        color = copy.deepcopy(color)
        color[i] = max(min(color[i] + random.randint(-min(offset, color[i]), min(offset, 255 - color[i])), 255), 0)
        return color



    def get_initial_solution(self):
        return [[self.get_random_polygon(), self.get_random_color()] for _ in range(self.num_shapes)]


    def get_random_color(self):
        return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


    def get_random_polygon(self):
        num_edges = random.randint(3, self.max_edges)
        return [self.get_random_point() for _ in range(num_edges)]


    def get_random_point(self):
        return random.randint(0, self.w), random.randint(0, self.h)


    def get_sol_diff(self, cand):
        sol = self.create_image_from_sol(cand)
        diff = self.target_image_diff - sol
        return diff


    def fitness(self, cand):
        start_time = time.time()
        diff = np.abs(self.get_sol_diff(cand)).sum()

        _fitness = (100 * diff / (self.w * self.h * 3 * 255)) ** 2

        self.fitness_time += (time.time() - start_time)
        self.fitness_count += 1

        return _fitness



    def create_image_from_sol(self, cand, to_rgb=True):
        sol = np.zeros((self.h, self.w, 4), np.uint8)

        for shape, color in cand:
            overlay = sol.copy()
            pts = np.array(shape)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.8, sol, 0.2, 0, sol)

        if to_rgb:
            sol = cv2.cvtColor(sol, cv2.COLOR_RGBA2RGB)
        return sol



    def plot_sol(self, sol):

        if self.draw_process is None:
            self.draw_process = multiprocessing.Process(None, self.__plot_sol, args=(self.draw_queue,))
            self.draw_process.start()

        if self.draw_queue.qsize() < 1:
            self.draw_queue.put(sol)



    def __plot_sol(self, queue):
        if not self.initilized_graph:
            plt.ion()
            plt.show()
            self.initilized_graph = True

        while True:
            try:
                sol = queue.get_nowait()

                if sol == 'Q':
                    plt.close()
                    return

                sol = self.create_image_from_sol(sol, True)
                cv2.imwrite(self.sol_file, sol)

                try:
                    im = np.concatenate((cv2.cvtColor(sol, cv2.COLOR_RGB2BGR), cv2.cvtColor(self.target_image, cv2.COLOR_RGB2BGR)), axis=1)
                except:
                    traceback.print_exc()

                plt.clf()
                plt.imshow(im)
                #plt.imshow(cv2.cvtColor(sol, cv2.COLOR_RGB2BGR))
                #plt.draw()
                plt.pause(0.0001)

            except:
                plt.pause(0.1)


    def finish(self):
        self.draw_queue.put('Q')


if __name__ == '__main__':
    print(datetime.now())
    img = cv2.imread('data/images/mona-lisa-head.png')
    improving_list = []

    def improve(caller):
        global improving_list
        improving_list.append(caller.best)
        problem.plot_sol(caller.best)

    initial_solution = None
    i = 1
    current_fitness = 10000000
    problem = PoligonProblem(img,
                             num_shapes=i,
                             candidates_by_iteration=100,
                             delta=50,
                             sol_file='data/images/mona-lisa-head-sol1.png',
                             max_edges=7,
                             vns_vnd='None')

    while initial_solution is None or len(initial_solution) < 100:
        problem.num_shapes = i
        # una posible mejora podria ser optimizar los polígonos que están introduciendo más error
        # i-1 optimiza el último poligono introducido
        problem.polygon_list = [i-1]

        if not initial_solution is None:
            initial_solution.append([problem.get_random_polygon(), problem.get_random_color()])

        #Ejecuta busqueda tabú
        searcher = TabuSearch(problem,
                              max_iterations=100,
                              list_length=2,
                              improved_event=improve,
                              tolerance=50)

        searcher.search(initial_solution=initial_solution)

        # Comprueba si la solucion actual es mejor que la encontrada
        if current_fitness > searcher.best_fitness or initial_solution is None:
            initial_solution = searcher.best
            current_fitness = searcher.best_fitness

            # aqui ya tenemos 2 vecinos
            if i > 1:
                # Indices de los poligonos que quiero optimizar
                # Con None le indico que lo aplique a todos los poligonos
                problem.polygon_list=None

                if problem.vns_vnd =='vnd':
                    print('VND search...')
                    vnd = VND(searcher, problem, ['move', 'color', 'add', 'remove'])
                    vnd.search(initial_solution)

                    initial_solution = vnd.best
                    current_fitness = vnd.best_fitness

                elif problem.vns_vnd =='vns':
                    print('VNS search...')
                    vns = VNS(searcher, problem, ['move', 'color', 'add', 'remove'])
                    vns.search(initial_solution)

                    initial_solution = vns.best
                    current_fitness = vns.best_fitness

                pass

            i += 1
            print("Solution length: %d" % i)

        else:
            initial_solution = initial_solution[:-1]

        print("num fitness per second: %f" % (problem.fitness_count / problem.fitness_time))


    print("General optimization")
    problem.polygon_list=None
    searcher = TabuSearch(problem,
                          max_iterations=10000,
                          list_length=100,
                          improved_event=improve)
    searcher.search(initial_solution=initial_solution)
    problem.finish()

    pk.dump(improving_list, open("result/mona_lisa_%f.pk" % searcher.best_fitness, "wb"))

    print(datetime.now())

    print("Finish")
