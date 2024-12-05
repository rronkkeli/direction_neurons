import numpy as np
# testipiste = np.array([[1308, 1622, 1669]])

def middle_point(point: np.ndarray) -> np.ndarray:
    x, y, z = point
    middle = None

    if (x <= y and x >= z) or (x >= y and x <= z):
        middle = 0
    elif (y <= x and y >= z) or (y >= x and y <= z):
        middle = 1
    elif (z <= x and y <= z) or (z >= x and y >= z):
        middle = 2

    return (point - point[middle])

# layer_data = middle_point(testipiste)

def output(point: np.ndarray) -> np.ndarray:
    weights = np.array([[-1, 1, 0, 0, 0, 0], [0, 0, -1, 1, 0, 0], [0, 0, 0, 0, -1, 1]])

    result = np.matmul(point, weights)

    print(np.argmax(result))

# output(layer_data)

def maximinimize(values: np.ndarray) -> np.ndarray:
    ix = np.argmax(values)
    flip = np.zeros((1, 6))
    flip[0, ix] = 1
    return flip
