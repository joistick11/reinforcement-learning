import numpy as np

V = [0, 0, 0]
for i in range(0, 3):
    V[i] = [0, 0, 0]
for i in range(0, 3):
    for j in range(0, 3):
        V[i][j] = [0, 0, 0, 0, 0]
for i in range(0, 3):
    for j in range(0, 3):
        for k in range(0, 5):
            V[i][j][k] = [0, 0, 0]
