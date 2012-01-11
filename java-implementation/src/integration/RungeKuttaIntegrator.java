package integration;

import java.io.IOException;
import java.io.PrintWriter;

public class RungeKuttaIntegrator
{
    private double _t;
    private double _stepSize;
    private boolean _printTime;

    public RungeKuttaIntegrator(double t, double stepSize, boolean printTime)
    {
        _t = t;
        _stepSize = stepSize;
        _printTime = printTime;
    }

    public RungeKuttaIntegrator(double t, double stepSize)
    {
        this(t, stepSize, true);
    }

    public void integrate(ODEGroup g, ResultsOutput out) throws IOException
    {
        double[] initialConditions = g.initialConditions();
        int numODEs = initialConditions.length;

        double[] k1 = new double[numODEs];
        double[] k2 = new double[numODEs];
        double[] k3 = new double[numODEs];
        double[] k4 = new double[numODEs];
        double[] yd = new double[numODEs];
        double[] dydx;

        // iteration over allowed steps
        double[] currentState = initialConditions;
        out.write(0, currentState);
        for(int step = 0; step * _stepSize <= _t; step++)
        {
            double t  = step * _stepSize;
            if (_printTime)
            {
                System.out.println(t);
            }
            dydx = g.calculateStep(currentState, t);
            for(int i=0; i<numODEs; i++)k1[i] = _stepSize*dydx[i];

            for(int i=0; i<numODEs; i++)yd[i] = currentState[i] + k1[i]/2;
            dydx = g.calculateStep(yd, t + _stepSize / 2);
            for(int i=0; i<numODEs; i++)k2[i] = _stepSize * dydx[i];

            for(int i=0; i<numODEs; i++)yd[i] = currentState[i] + k2[i]/2;
            dydx = g.calculateStep(yd, t + _stepSize / 2);
            for(int i=0; i<numODEs; i++)k3[i] = _stepSize * dydx[i];

            for(int i=0; i<numODEs; i++)yd[i] = currentState[i] + k3[i];
            dydx = g.calculateStep(yd, t + _stepSize);
            for(int i=0; i<numODEs; i++)k4[i] = _stepSize * dydx[i];

            double[] y =new double[numODEs];
            for(int i=0; i<numODEs; i++)y[i] = currentState[i] + k1[i]/6 + k2[i]/3 + k3[i]/3 + k4[i]/6;
            currentState = y;
            out.write(t, currentState);
        }
        out.complete();
    }
}
