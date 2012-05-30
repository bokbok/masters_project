from pylab import plot, show, figure, draw
from mpl_toolkits.mplot3d import Axes3D
from numpy import ones

class Display:

    def __init__(self, x, y, z, fig = "7", xrange = 0.1, yrange = 0.1, zrange = 0.1):
        self.figr = figure(fig)
        self.axes = Axes3D(self.figr)
        self.axes.legend()
        self.axes.set_xlabel(x)
        self.axes.set_ylabel(y)
        self.axes.set_zlabel(z)

        self.x = x
        self.y = y
        self.z = z
        self.xrange = [-xrange, xrange]
        self.yrange = [-yrange, yrange]
        self.zrange = [-zrange, zrange]

    def addSliceMinMaxZ(self, continuation, zVal):
        xSolMin = continuation.vals(self.x + "_min")
        xSolMax = continuation.vals(self.x + "_max")

        ySol = continuation.vals(self.y)

        z = ones(len(ySol)) * zVal

        self.axes.plot(xSolMin, ySol, z, label='3D')
        self.axes.plot(xSolMax, ySol, z, label='3D')
        self.setLimits()
        return self

    def addSliceY(self, continuation, yVal):
        xSol = continuation.vals(self.x)

        zSol = continuation.vals(self.z)

        y = ones(len(zSol)) * yVal

        self.axes.plot(xSol, y, zSol, label='3D')
        self.setLimits()
        return self

    def addSliceZ(self, continuation, zVal):
        xSol = continuation.vals(self.x)

        ySol = continuation.vals(self.y)

        z = ones(len(ySol)) * zVal

        self.axes.plot(xSol, ySol, z, label='3D')
        self.setLimits()
        return self


    def addSliceMinMaxY(self, continuation, yVal):
        xSolMin = continuation.vals(self.x + "_min")
        xSolMax = continuation.vals(self.x + "_max")

        zSol = continuation.vals(self.z)

        y = ones(len(zSol)) * yVal

        self.axes.plot(xSolMin, y, zSol, label='3D')
        self.axes.plot(xSolMax, y, zSol, label='3D')
        self.setLimits()
        return self

    def add3DPhasePlane(self, run):
        self.axes.plot(run.vals(self.x), run.vals(self.y), run.vals(self.z))

    def setLimits(self):
        draw()
        self.axes.set_xlim3d(self.xrange)
        self.axes.set_ylim3d(self.yrange)
        self.axes.set_zlim3d(self.zrange[0], self.zrange[1])



