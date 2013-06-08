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

//#define _RK_DEBUG
#ifdef _RK_DEBUG
	#define NAN_CHECK(s, name, previous) \
			if (s.debugNanCheck()) \
			{ \
				printf("%s NAN (x, y): %i, %i\n",name, previous.x(), previous.y()); \
			}
	#define DUMP(s, name, xv, yv) \
		if (_previous.x() == xv && _previous.y() == yv) { \
			for (int i = 0; i < MAX_EQUATIONS; i++) \
			{ \
				printf("%s % i %f (x, y): %i, %i\n", name, i, s[i], _previous.x(), _previous.y()); \
			} }

#else
	#define NAN_CHECK(s, name, previous)
	#define DUMP(s, name, xv, yv)
#endif // RK_DEBUG



template <class T>
class RungeKuttaIntegrator
{
public:
	__device__
	RungeKuttaIntegrator(DeviceMeshPoint curr, DeviceMeshPoint prev, double t, double deltaT, double delta):
		_current(curr), _previous(prev), _t(t), _deltaT(deltaT), _deltaSpatial(delta)
	{
	}

	/////////////////////////////////
	//        f=F(x0,y0);
	//        k1 = h * f;
	//        f = F(x0+h/2,y0+k1/2);
	//        k2 = h * f;
	//        f = F(x0+h/2,y0+k2/2);
	//        k3 = h * f;
	//        f = F(x0+h/2,y0+k3/2);
	//        k4 = h * f;
	//////////////////////////////////
	__device__
	void integrateStep()
	{
		StateSpace prev = _previous.state();
		ParameterSpace params = _previous.parameters();

		Derivatives f1 = _model(prev, params, _previous);
		StateSpace k1 = f1 * _deltaT;

		StateSpace k1Div2 = (k1 / 2.0);

		StateSpace f2Pt = prev + k1Div2;

		Derivatives f2 = _model(f2Pt, params, _previous);
		StateSpace k2 = f2 * _deltaT;


		StateSpace k2Div2 = (k1 / 2.0);

		StateSpace f3Pt = prev + k2Div2;

		Derivatives f3 = _model(f3Pt, params, _previous);

		StateSpace k3 = f3 * _deltaT;

		StateSpace k3Div2 = (k1 / 2.0);

		StateSpace f4Pt = prev + k3Div2;

		Derivatives f4 = _model(f4Pt, params, _previous);
		StateSpace k4 = f4 * _deltaT;


		StateSpace k2Mul2 = k2 * 2.0;

		StateSpace k3Mul2 = k3 * 2.0;

		StateSpace add1 = k1 + k2Mul2;

		StateSpace add2 = k3Mul2 + k4;

		StateSpace add = add1 + add2;

		StateSpace div6 = add / 6;

		StateSpace integrated = prev + div6;

		_current.state().update(_t, integrated);
	}


private:
	DeviceMeshPoint _current, _previous;
	double _t, _deltaT, _deltaSpatial;

	T _model;
};

#endif /* RUNGEKUTTAINTEGRATOR_CUH_ */
