package model;

import integration.RungeKuttaIntegrator;
import integration.SimpleResultsOutput;
import org.junit.Test;

import java.io.*;

import static org.junit.Assert.assertEquals;

public class SolutionTest
{
    @Test
    public void itShouldRunAndLookOKComparedToAWellKnownRun() throws IOException
    {
        SimpleResultsOutput out = new SimpleResultsOutput();
        Solution solution = new Solution(-70, -70, 300, 65,
                                         3034, 3034,
                                         536, 536,
                                         4000, 2000,
                                         0.4, 0.4, 0.11, 0.01, 700,
                                         0.4, 0.8,
                                         0, 0, 0, 0,
                                         500, 500, -50, -50,
                                         5, 5, 45, -90, 50, 50, new ByteArrayOutputStream());

        RungeKuttaIntegrator integrator = new RungeKuttaIntegrator(0.01, 1e-5, false);
        integrator.integrate(solution, out);

        checkResults(out);

    }

    private void checkResults(SimpleResultsOutput out) throws IOException
    {
        BufferedReader file = new BufferedReader(new InputStreamReader(SolutionTest.class.getResourceAsStream("./test.dat")));


        String line = file.readLine();
        while((line = file.readLine()) != null)
        {
            String[] values = line.split(":");
            double t = Double.parseDouble(values[0]);

            double[] simVals = out.get(t);

            for (int i = 1; i < values.length; i++)
            {
                assertEquals("t = " + t + ", i = " + i, Double.parseDouble(values[i]), simVals[i - 1], 0.001);
            }
        }
    }

}
