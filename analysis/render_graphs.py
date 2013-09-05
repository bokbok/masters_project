from binary_reader import Reader, HeaderInfo
import sys
import matplotlib.pyplot as plt
import ntpath

dataFile = sys.argv[1]
tMax = float(sys.argv[2])

pointNum = 3

points = []

baseDir = ntpath.dirname(dataFile)

header = HeaderInfo(dataFile)

while len(sys.argv) > pointNum:
    point = sys.argv[pointNum].split(',')
    points.append((int(point[0]), int(point[1])))
    pointNum += 1


for dim in header.dims:
    print "Rendering " + dim
    reader = Reader(sys.argv[1], dim, points, tMax)

    data = reader.readAll()

    plt.figure()
    for point in points:
        plt.plot(data['time'], data['data'][point], label = "(" + str(point[0]) + ", " + str(point[1]) + ")")

    plt.legend()
    plt.ylabel(dim)
    plt.xlabel('t')
    plt.savefig(baseDir + "/" + dim + '.pdf')