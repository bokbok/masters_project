from binary_reader import Reader
import sys
import matplotlib.pyplot as plt

tMax = float(sys.argv[4])
dim = sys.argv[3]
reader = Reader(sys.argv[1], int(sys.argv[2]))


pointNum = 5

points = []

while len(sys.argv) > pointNum:
    point = sys.argv[pointNum].split(',')
    points.append((int(point[0]), int(point[1])))
    pointNum += 1


data = reader.read_points_dim(dim, points, tMax)

for point in points:
    plt.plot(data['time'], data['data'][point], label = "(" + str(point[0]) + ", " + str(point[1]) + ")")

plt.legend()
plt.ylabel(dim)
plt.xlabel('t')

plt.show()
