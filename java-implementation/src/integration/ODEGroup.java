package integration;

public interface ODEGroup
{
    double[] calculateStep(double[] state, double t);
    double[] initialConditions();
}
