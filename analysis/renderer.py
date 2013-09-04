import matplotlib.pyplot as plt
from matplotlib import animation
from numpy  import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Renderer:
    def __init__(self, reader, fileOut, gridSize, step, tMax):
        self.reader = reader
        self.fileOut = fileOut
        self.fig = plt.figure(figsize = (10, 10))
        self.axes = Axes3D(self.fig)
        self.arraySize = int(gridSize / step)
        self.gridSize = gridSize
        self.step = step
        self.axes.set_zlim3d(-80, -50)
        self.tMax = tMax
        self.fps = 30
        self.totalFrames = int(self.fps * tMax)

    def render(self):
        drawCallback = (lambda frame: self.draw())
        self.anim = animation.FuncAnimation(self.fig, drawCallback,  frames=self.totalFrames, interval=1, blit=False)
        self.anim.save(self.fileOut, fps=self.fps)

    def draw(self):
        t, data = self.reader.readNextFrame()

        Z = arange(self.arraySize ** 2).reshape(self.arraySize, self.arraySize)

        X = arange(0, self.gridSize, self.step)
        Y = arange(0, self.gridSize, self.step)
        X, Y = meshgrid(X, Y)

        if data != None:
            for point in data.keys():
                xIndex = int(float(point[0]) / float(self.step))
                yIndex = int(float(point[1]) / float(self.step))
                Z[xIndex][yIndex] = data[point]

            self.axes.cla()
            self.axes.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                   cmap=cm.coolwarm,
                                   linewidth=0,
                                   antialiased=False)
            self.axes.set_zlim3d(-80, 0)
            self.axes.text(2, 6, 1, 't=' + str(t), fontsize=15)

            print "Plot - " + str(t)
        return self.axes







