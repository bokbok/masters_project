/*
 * RungeKuttaIntegrator.cuh
 *
 *  Created on: 01/05/2013
 *      Author: matt
 */

#ifndef RUNGEKUTTAINTEGRATOR_CUH_
#define RUNGEKUTTAINTEGRATOR_CUH_

#include "DeviceMeshPoint.cuh"
#include "StateSpace.cuh"
#include "Derivatives.cuh"
#include <stdio.h>

template <class T>
class RungeKuttaIntegrator
{
public:
	__device__
	RungeKuttaIntegrator(DeviceMeshPoint prev, DeviceMeshPoint curr, double t, double deltaT):
		_current(curr), _previous(prev), _t(t), _deltaT(deltaT)
	{
	}

	__device__
	void integrateStep()
	{
		printf("Integrating %f\n", _t);

		Derivatives f1 = _model(_previous.state(), _previous);
		StateSpace k1 = f1 * _deltaT;

		StateSpace & prev = _previous.state();

		StateSpace k1Div2 = (k1 / 2.0);
		StateSpace f2Pt = prev + k1Div2;

		Derivatives f2 = _model(f2Pt, _previous);
		StateSpace k2 = f2 * _deltaT;

		StateSpace k2Div2 = (k1 / 2.0);
		StateSpace f3Pt = prev + k2Div2;
		Derivatives f3 = _model(f3Pt, _previous);

		StateSpace k3 = f3 * _deltaT;

		StateSpace k3Div2 = (k1 / 2.0);
		StateSpace f4Pt = prev + k3Div2;
		Derivatives f4 = _model(f4Pt, _previous);
		StateSpace k4 = f4 * _deltaT;

		StateSpace k2Mul2 = k2 * 2.0;
		StateSpace k3Mul2 = k3 * 2.0;



		StateSpace add1 = k1 + k2Mul2;
		StateSpace add2 = k3Mul2 + k4;
		StateSpace add = add1 + add2;
		StateSpace div6 = add / 6;
		StateSpace integrated = prev + div6;

		_current.state().update(integrated);

//        f=F(x0,y0);
//        k1 = h * f;
//        f = F(x0+h/2,y0+k1/2);
//        k2 = h * f;
//        f = F(x0+h/2,y0+k2/2);
//        k3 = h * f;
//        f = F(x0+h/2,y0+k2/2);
//        k4 = h * f;

	}

private:
	DeviceMeshPoint _current, _previous;
	double _t, _deltaT;

	T _model;
};

#endif /* RUNGEKUTTAINTEGRATOR_CUH_ */
