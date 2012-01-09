package integration;

import java.io.IOException;

public interface ResultsOutput
{
    void write(double t, double[] currentState) throws IOException;
    void complete();
}
