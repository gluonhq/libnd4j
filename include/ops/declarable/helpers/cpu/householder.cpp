//
// Created by Yurii Shyrma on 18.12.2017
//

#include <ops/declarable/CustomOperations.h>
#include <DataTypeUtils.h>
#include <ops.h>
#include <ops/declarable/helpers/householder.h>

namespace nd4j {
namespace ops {
namespace helpers {



//////////////////////////////////////////////////////////////////////////
template <typename T>
void evalHouseholderData(const NDArray<T>& x, NDArray<T>& tail, T& normX, T& coeff) {

	// input validation
	if(!x.isVector())
		throw "ops::helpers::houseHolderForVector function: input array must be vector !";
	
	if(!tail.isVector())
		throw "ops::helpers::houseHolderForVector function: output array must be vector !";

	if(x.lengthOf() != tail.lengthOf() + 1)
		throw "ops::helpers::houseHolderForVector function: output vector must have length smaller by unity compared to input vector !";
		
	normX = x.template reduceNumber<simdOps::Norm2<T>>();	
	const T min = DataTypeUtils::min<T>();

	if(normX*normX - x(0)*x(0) <= min) {

		coeff = (T)0.;
		normX = x(0); 
		tail = (T)0.;
	}
	else {
		
		if(x(0) >= (T)0.)
			normX = -normX;									// choose opposite sign to lessen roundoff error
		const T u0 = x(0) - normX;
		coeff = -u0 / normX;
		
		if(x.isRowVector())
			tail = x({{}, {1,-1}}) / u0;
		else
			tail = x({{1,-1}, {}}) / u0;
	} 
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T> evalHouseholderMatrix(const NDArray<T>& x) {

	// input validation
	if(!x.isVector() && !x.isScalar())
		throw "ops::helpers::evalHouseholderMatrix function: input array must be vector or scalar!";
			
	NDArray<T> w((int)x.lengthOf(), 1,  x.ordering(), x.getWorkspace());							// column-vector
	NDArray<T> wT(1, (int)x.lengthOf(), x.ordering(), x.getWorkspace());							// row-vector (transposed w)	

	T normX = x.template reduceNumber<simdOps::Norm2<T>>();	
	const T min = DataTypeUtils::min<T>();
	T coeff;

	if(x.isScalar() || normX*normX - x(0)*x(0) <= min) {

		coeff = (T)0.;
		normX = x(0); 
		w = (T)0.;
	}
	else {
		
		if(x(0) >= (T)0.)
			normX = -normX;									// choose opposite sign to lessen roundoff error
		
		T u0 = x(0) - normX;
		coeff = -u0 / normX;				
		w.assign(x / u0);
	}

	w(0) = (T)1.;
	wT.assign(w);

	NDArray<T> identity((int)x.lengthOf(), (int)x.lengthOf(), x.ordering(), x.getWorkspace());					 
	identity.setIdentity();																			// identity matrix	

	return (identity - mmul(w, wT) * coeff);
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void biDiagonalizeUp(NDArray<T>& matrix) {

	// input validation
	if(matrix.rankOf() != 2 || matrix.isScalar())
		throw "ops::helpers::biDiagonalizeUp function: input array must be 2D matrix !";

	const int rows = matrix.sizeAt(0);
	const int cols = matrix.sizeAt(1);
	if(rows < cols)
		throw "ops::helpers::biDiagonalizeUp function: this procedure is applicable for input matrix with rows >= cols !";
		
	NDArray<T>* bottomRightCornerOfMatrix = nullptr;
	for(int i = 0; i < cols-1; ++i ) {

		// evaluate Householder matrix acting on columns
		NDArray<T> Pcols = evalHouseholderMatrix(matrix({{i, rows}, {i, i+1}}));
		// Pcols * bottomRightCornerOfMatrix - multiply given matrix block on householder matrix from the left
		// matrix({{i, rows}, {i, cols-1}}) = mmul(Pcols, matrix({{i, rows}, {i, cols-1}}));
		bottomRightCornerOfMatrix =  matrix.subarray({{i, rows}, {i, cols}});
		NDArrayFactory<T>::mmulHelper(&Pcols, bottomRightCornerOfMatrix, bottomRightCornerOfMatrix, (T)1., (T)0.);
		delete bottomRightCornerOfMatrix;

		if(i == cols-2) continue; 							// do not apply last iteration to rows
			
		// evaluate Householder matrix acting on rows
		NDArray<T> Prows = evalHouseholderMatrix(matrix({{i+1, i+2}, {i+1, cols}}));
		// bottomRightCornerOfMatrix * Prols - multiply given matrix block on householder matrix from the right		
		bottomRightCornerOfMatrix = matrix.subarray({{i+1, rows}, {i+1, cols}});
		NDArrayFactory<T>::mmulHelper(bottomRightCornerOfMatrix, &Prows, bottomRightCornerOfMatrix, (T)1., (T)0.);
		delete bottomRightCornerOfMatrix;
	}

	
}



template void evalHouseholderData<float>  (const NDArray<float  >& x, NDArray<float  >& tail, float  & normX, float  & coeff);
template void evalHouseholderData<float16>(const NDArray<float16>& x, NDArray<float16>& tail, float16& normX, float16& coeff);
template void evalHouseholderData<double> (const NDArray<double >& x, NDArray<double >& tail, double & normX, double & coeff);

template NDArray<float>   evalHouseholderMatrix<float>  (const NDArray<float  >& x);
template NDArray<float16> evalHouseholderMatrix<float16>(const NDArray<float16>& x);
template NDArray<double>  evalHouseholderMatrix<double> (const NDArray<double >& x);

template void biDiagonalizeUp<float>  (NDArray<float  >& matrix);
template void biDiagonalizeUp<float16>(NDArray<float16>& matrix);
template void biDiagonalizeUp<double> (NDArray<double >& matrix);

}
}
}