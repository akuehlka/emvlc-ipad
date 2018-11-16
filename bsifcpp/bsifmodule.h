/*
 * bsifmodule.h
 *
 *  Created on: Jul 13, 2017
 *      Author: andrey
 */

#ifndef BSIFMODULE_H_
#define BSIFMODULE_H_

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <vector>

boost::python::numpy::ndarray
bsif_extract(boost::python::numpy::ndarray src,
		boost::python::numpy::ndarray& dst,
		boost::python::numpy::ndarray filterdims);

#endif /* BSIFMODULE_H_ */
