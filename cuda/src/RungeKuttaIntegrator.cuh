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
#include <stdio.h>
#include "common.cuh"

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
	RungeKuttaIntegrator(DeviceMeshPoint curr, DeviceMeshPoint prev, double t, double deltaT):
		_current(curr), _previous(prev), _t(t), _deltaT(deltaT)
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
		double state[MAX_EQUATIONS];
		double derivatives[MAX_EQUATIONS];
		double integrated[MAX_EQUATIONS];

		StateSpace & prev = _previous.state();
		ParameterSpace & params = _previous.parameters();

		_model(derivatives, prev, params, _previous);

		for (int i = 0; i < prev.numDimensions(); i++)
		{
			double dy1 = derivatives[i] * _deltaT;
			integrated[i] = prev[i] + dy1 / 6;
			state[i] = prev[i] + dy1 / 2.0;
		}

		_model(derivatives, state, params, _previous);
		for (int i = 0; i < prev.numDimensions(); i++)
		{
			double dy2 = derivatives[i] * _deltaT;
			integrated[i] += dy2 / 3;
			state[i] = prev[i] + dy2 / 2.0;
		}

		_model(derivatives, state, params, _previous);
		for (int i = 0; i < prev.numDimensions(); i++)
		{
			double dy3 = derivatives[i] * _deltaT;
			integrated[i] += dy3 / 3;
			state[i] = prev[i] + dy3;
		}
		_model(derivatives, state, params, _previous);
		for (int i = 0; i < prev.numDimensions(); i++)
		{
			double dy4 = derivatives[i] * _deltaT;
			integrated[i] += dy4 / 6;
		}

		_current.state().update(_t, integrated);
	}


private:
	DeviceMeshPoint & _current, _previous;
	double _t, _deltaT;

	T _model;
};

#endif /* RUNGEKUTTAINTEGRATOR_CUH_ */
