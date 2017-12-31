//
// Created by Yurii Shyrma on 18.12.2017
//

// #include <ops/declarable/CustomOperations.h>
// #include <DataTypeUtils.h>
// #include <ops.h>
#include <ops/declarable/helpers/householder.h>

namespace nd4j {
namespace ops {
namespace helpers {



//////////////////////////////////////////////////////////////////////////
// template <typename T>
// void evalHouseholderData(const NDArray<T>& x, NDArray<T>& tail, T& normX, T& coeff) {

// 	// input validation
// 	if(!x.isVector())
// 		throw "ops::helpers::houseHolderForVector function: input array must be vector !";
	
// 	if(!tail.isVector())
// 		throw "ops::helpers::houseHolderForVector function: output array must be vector !";

// 	if(x.lengthOf() != tail.lengthOf() + 1)
// 		throw "ops::helpers::houseHolderForVector function: output vector must have length smaller by unity compared to input vector !";
		
// 	normX = x.template reduceNumber<simdOps::Norm2<T>>();	
// 	const T min = DataTypeUtils::min<T>();

// 	if(normX*normX - x(0)*x(0) <= min) {

// 		coeff = (T)0.;
// 		normX = x(0); 
// 		tail = (T)0.;
// 	}
// 	else {
		
// 		if(x(0) >= (T)0.)
// 			normX = -normX;									// choose opposite sign to lessen roundoff error
// 		const T u0 = x(0) - normX;
// 		coeff = -u0 / normX;
		
// 		if(x.isRowVector())
// 			tail = x({{}, {1,-1}}) / u0;
// 		else
// 			tail = x({{1,-1}, {}}) / u0;
// 	} 
// }

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T> Householder<T>::evalHHmatrix(NDArray<T>& x, T& coeff, T& normX) {

	// input validation
	if(!x.isVector() && !x.isScalar())
		throw "ops::helpers::evalHHmatrix function: input array must be vector or scalar!";

	// NDArray<T> w((int)x.lengthOf(), 1,  x.ordering(), x.getWorkspace());							// column-vector
	// NDArray<T> wT(1, (int)x.lengthOf(), x.ordering(), x.getWorkspace());							// row-vector (transposed w)	

	normX = x.template reduceNumber<simdOps::Norm2<T>>();	
	const T min = DataTypeUtils::min<T>();	
	
	if (x.isScalar()) {
		
		NDArray<T> result(1, 1, x.ordering(), x.getWorkspace());
		result(0) = (T)1. - coeff;	

		return result;
	}
	else if(normX*normX - x(0)*x(0) <= min) {

		coeff = (T)0.;
		normX = x(0); 
		x.assign((T)0.);
		
	} 	
	else {
		
		if(x(0) >= (T)0.)
			normX = -normX;									// choose opposite sign to lessen roundoff error
		
		T u0 = x(0) - normX;
		coeff = -u0 / normX;				
		x.assign(x / u0);
	}
	
	x(0) = (T)1.;
	
	NDArray<T> identity((int)x.lengthOf(), (int)x.lengthOf(), x.ordering(), x.getWorkspace());					 
	identity.setIdentity();																			// identity matrix	
		
	NDArray<T>* xT = x.transpose();
	NDArray<T> *pRow(nullptr), *pCol(nullptr);
	
	if(x.isRowVector()) {
		
		pRow = &x;
		pCol = xT;
	}
	else {
	
		pRow = xT;
		pCol = &x;	
	}

	NDArray<T> result = identity - mmul(*pCol, *pRow) * coeff;	
	delete xT;
	
	return result;
}


template class ND4J_EXPORT Householder<float>;
template class ND4J_EXPORT Householder<float16>;
template class ND4J_EXPORT Householder<double>;







}
}
}