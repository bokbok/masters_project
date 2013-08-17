from binary_reader import Reader
import sys
import matplotlib.pyplot as plt

tMax = float(sys.argv[3])
dim = sys.argv[2]

pointNum = 4

points = []

while len(sys.argv) > pointNum:
    point = sys.argv[pointNum].split(',')
    points.append((int(point[0]), int(point[1])))
    pointNum += 1

reader = Reader(sys.argv[1], dim, points, tMax)


data = reader.readAll()

for point in points:
    plt.plot(data['time'], data['data'][point], label = "(" + str(point[0]) + ", " + str(point[1]) + ")")

plt.legend()
plt.ylabel(dim)
plt.xlabel('t')

plt.show()
