package integration;

import org.junit.Test;

import java.io.IOException;

import static java.lang.Math.*;
import static org.junit.Assert.*;

public class RungeKutaIntegratorTest
{
    @Test
    public void itShouldIntegrateToAnExpSolutionFOrFirstOrder() throws IOException
    {
        SimpleResultsOutput out = new SimpleResultsOutput();
        new RungeKuttaIntegrator(10, 1e-4, false).integrate(new FirstOrderODE(), out);

        assertEquals(4f, out.get(0)[0], 0.01);
        assertEquals(pow(2, 2 - 10), out.get(10)[0], 0.01);
        assertEquals(pow(2, 2 - 5), out.get(5)[0], 0.01);
    }

    @Test
    public void itShouldIntegrateYEqualsXpow2() throws IOException
    {
        SimpleResultsOutput out = new SimpleResultsOutput();
        new RungeKuttaIntegrator(10, 1e-4, false).integrate(new X2(), out);

        assertEquals(0.33333333 * pow(1, 3), out.get(1)[0], 0.01);
        assertEquals(0.33333333 * pow(5, 3), out.get(5)[0], 0.01);
        assertEquals(0.33333333 * pow(7, 3), out.get(7)[0], 0.01);
    }

    private static class FirstOrderODE implements ODEGroup
    {

        public double[] calculateStep(double[] state, double t)
        {
            // dy/dt = -alpha y : y(1)=2, y(2)=1, alpha = ln(2), analytic solution y = 2^2 - t
            return new double[]{ -log(2) * state[0] };
        }

        public double[] initialConditions()
        {
            return new double[]{ 4 };
        }
    }

    private static class X2 implements ODEGroup
    {

        public double[] calculateStep(double[] state, double t)
        {
            return new double[]{ pow(t, 2) };
        }

        public double[] initialConditions()
        {
            return new double[]{ 0 };
        }
    }
}
