from math import cos, sqrt, tan, radians
from typing import Callable, Set, Tuple, TypedDict
from scipy import optimize
import time

# from src.sampling_complete_data import PosData


def _generate_func(data) -> Callable:
    theta = data['angle'][0]
    phi = data['angle'][1]
    psi = data['angle'][2]
    alpha = data['grad'][0]
    beta = data['grad'][1]
    a = data['a']

    def func(x):
        return [((x[0]-a)*tan(alpha) + x[1]*tan(beta) - x[2])
                / (sqrt(1 + tan(alpha)**2 + tan(beta)**2)
                   * sqrt((x[0]-a)**2 + x[1]**2 + x[2]**2)) + cos(theta),
                ((x[0]+a)*tan(alpha) + x[1]*tan(beta) - x[2])
                / (sqrt(1 + tan(alpha) ** 2 + tan(beta)**2)
                   * sqrt((x[0]+a)**2 + x[1]**2 + x[2]**2)) + cos(phi),
                ((x[0]*tan(beta)+x[1])**2 + (x[2]*tan(alpha)+x[0])**2 - a**2
                 + (x[1]*tan(alpha)-x[0]*tan(beta))**2 - a**2*tan(beta)**2)
                / (sqrt((x[2]*tan(beta)+x[1])**2 + (x[2]*tan(alpha)+x[0]-a)**2
                        + (x[1]*tan(alpha)-(x[0]-a)*tan(beta))**2)
                   * sqrt((x[2]*tan(beta)+x[1])**2
                          + (x[2]*tan(alpha)+x[0]+a)**2
                          + (x[1]*tan(alpha)-(x[0]+a)*tan(beta))**2))
                - cos(psi)]
    return func


def solve_nonlin(data, includeWrong: bool = False):
    root_set = set()
    initial_time = time.time()

    for init_x in range(-30, 30, 3):
        print("init_x:", init_x)
        for init_y in range(-30, 30, 3):
            for init_z in range(0, 30, 3):
                if any([(init_x, init_y, init_z) == (data['a'], 0, 0),
                       (init_x, init_y, init_z) == (-data['a'], 0, 0)]):
                    continue

                result = optimize.root(
                    _generate_func(data), [init_x, init_y, init_z],
                    method='broyden1',
                    options={
                        "maxiter": 30
                    })
                print(result['x'], result['success'])
                if(result['success']):
                    return result['x']

