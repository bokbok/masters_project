package integration;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class SimpleResultsOutput implements ResultsOutput
{
    private static Map<Double, double[]> _results = new HashMap<Double, double[]>();

    public double[] get(double t)
    {
        return _results.get(t);
    }

    public void write(double t, double[] currentState) throws IOException
    {
        _results.put(t, currentState);
    }

    public void complete()
    {

    }
}
