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
		StateSpace prev = _previous.state();
		ParameterSpace params = _previous.parameters();

		NAN_CHECK(prev, "prev", _previous);
		DUMP(prev, "prev", 0, 0);

		Derivatives f1 = _model(prev, params, _previous);
		StateSpace k1 = f1 * _deltaT;
		DUMP(f1, "f1", 0, 0);
		DUMP(k1, "k1", 0, 0);

		NAN_CHECK(f1, "f1", _previous);
		NAN_CHECK(k1, "k1", _previous);

		StateSpace k1Div2 = (k1 / 2.0);
		NAN_CHECK(k1Div2, "k1Div2", _previous);

		StateSpace f2Pt = prev + k1Div2;
		NAN_CHECK(f2Pt, "f2Pt", _previous);

		Derivatives f2 = _model(f2Pt, params, _previous);
		StateSpace k2 = f2 * _deltaT;
		NAN_CHECK(f2, "f2", _previous);
		NAN_CHECK(k2, "k2", _previous);


		StateSpace k2Div2 = (k1 / 2.0);
		NAN_CHECK(k2Div2, "k2Div2", _previous);

		StateSpace f3Pt = prev + k2Div2;
		NAN_CHECK(f3Pt, "f3Pt", _previous);

		Derivatives f3 = _model(f3Pt, params, _previous);

		StateSpace k3 = f3 * _deltaT;
		NAN_CHECK(f3, "f3", _previous);
		NAN_CHECK(k3, "k3", _previous);

		StateSpace k3Div2 = (k1 / 2.0);
		NAN_CHECK(k3Div2, "k3Div2", _previous);

		StateSpace f4Pt = prev + k3Div2;
		NAN_CHECK(f4Pt, "f4Pt", _previous);

		Derivatives f4 = _model(f4Pt, params, _previous);
		NAN_CHECK(f4, "f4", _previous);
		StateSpace k4 = f4 * _deltaT;
		NAN_CHECK(k4, "k4", _previous);


		StateSpace k2Mul2 = k2 * 2.0;
		NAN_CHECK(k2Mul2, "k2Mul2", _previous);

		StateSpace k3Mul2 = k3 * 2.0;
		NAN_CHECK(k3Mul2, "k3Mul2", _previous);

		StateSpace add1 = k1 + k2Mul2;
		NAN_CHECK(add1, "add1", _previous);

		StateSpace add2 = k3Mul2 + k4;
		NAN_CHECK(add2, "add2", _previous);

		StateSpace add = add1 + add2;
		NAN_CHECK(add, "add", _previous);

		StateSpace div6 = add / 6;
		NAN_CHECK(div6, "div6", _previous);

		StateSpace integrated = prev + div6;
		NAN_CHECK(integrated, "integrated", _previous);

		_current.state().update(integrated);
	}


private:
	DeviceMeshPoint _current, _previous;
	double _t, _deltaT;

	T _model;
};

#endif /* RUNGEKUTTAINTEGRATOR_CUH_ */
