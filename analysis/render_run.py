from binary_reader import Reader
from renderer import Renderer
import sys
import ntpath


tMax = float(sys.argv[4])
dim = sys.argv[3]
gridSize = int(sys.argv[2])
filename = sys.argv[1]
step = int(sys.argv[5])

arraySize = int(gridSize / step)

points = []

for x in range(0, gridSize, step):
    for y in range(0, gridSize, step):
        points.append((x, y))

reader = Reader(filename, dim, points, tMax, skip = 10)

renderer = Renderer(reader, ntpath.dirname(filename) + '/render.mp4', gridSize, step, tMax)

renderer.render()
