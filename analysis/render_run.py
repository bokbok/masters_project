from binary_reader import Reader
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy  import *
from mpl_toolkits.mplot3d import Axes3D
import ntpath


from matplotlib import animation

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

reader = Reader(sys.argv[1], dim, points, tMax, skip = 500)

fig = plt.figure(figsize = (10, 10))
ax = Axes3D(fig)

def draw(frame):
    t, data = reader.readNextFrame()

    Z = arange(arraySize ** 2).reshape(arraySize, arraySize)

    X = arange(0, gridSize, step)
    Y = arange(0, gridSize, step)
    X, Y = meshgrid(X, Y)

    if data != None:
        for point in data.keys():
            xIndex = int(float(point[0]) / float(step))
            yIndex = int(float(point[1]) / float(step))
            Z[xIndex][yIndex] = data[point]

        ax.cla()
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='green',
                               linewidth=0, antialiased=False, shade = True)
        ax.set_zlim3d(-80, 0)

        print "Plot - " + str(t)
    return ax

def init():
    ax.set_zlim3d(-80, -50)


anim = animation.FuncAnimation(fig, draw, init_func=init,
                               frames=int(tMax * 10), interval=1, blit=False)

anim.save('tmp/' + ntpath.basename(sys.argv[1]) + '.mp4', fps=10)
#plt.show()


