//
// Created by Yurii Shyrma on 18.12.2017
//

#include <ops/declarable/helpers/householder.h>
#include <ops/declarable/helpers/biDiagonalUp.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
BiDiagonalUp<T>::BiDiagonalUp(const NDArray<T>& matrix): _HHmatrix(NDArray<T>(matrix.sizeAt(0), matrix.sizeAt(1), matrix.ordering(), matrix.getWorkspace())),  
                                                         _HHbidiag(NDArray<T>(matrix.sizeAt(1), matrix.sizeAt(1), matrix.ordering(), matrix.getWorkspace())) {

	// input validation
	if(matrix.rankOf() != 2 || matrix.isScalar())
		throw "ops::helpers::biDiagonalizeUp constructor: input array must be 2D matrix !";

	_HHmatrix.assign(&matrix);
	
	evalData();

}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void BiDiagonalUp<T>::evalData() {
	
	const int rows = _HHmatrix.sizeAt(0);
	const int cols = _HHmatrix.sizeAt(1);
	
	if(rows < cols)
		throw "ops::helpers::BiDiagonalizeUp::evalData method: this procedure is applicable only for input matrix with rows >= cols !";
		
	NDArray<T>* bottomRightCornerOfMatrix = nullptr;
	for(int i = 0; i < cols-1; ++i ) {

		// evaluate Householder matrix nullifying  columns 
		NDArray<T> Pcols = Householder<T>::evalHHmatrix(_HHmatrix({{i, rows}, {i, i+1}}), _HHmatrix(i,i), _HHbidiag(i,i));
		// multiply given matrix block on householder matrix from the left: Pcols * bottomRightCornerOfMatrix
		bottomRightCornerOfMatrix =  _HHmatrix.subarray({{i, rows}, {i+1, cols}});	// {i, cols}
		NDArrayFactory<T>::mmulHelper(&Pcols, bottomRightCornerOfMatrix, bottomRightCornerOfMatrix, (T)1., (T)0.);
		delete bottomRightCornerOfMatrix;

		if(i == cols-2) continue; 							// do not apply right multiplying on last iteration

		// evaluate Householder matrix nullifying rows 
		NDArray<T> Prows = Householder<T>::evalHHmatrix(_HHmatrix({{i, i+1}, {i+1, cols}}), _HHmatrix(i,i+1), _HHbidiag(i,i+1));		
		// multiply given matrix block on householder matrix from the right: bottomRightCornerOfMatrix * Prols
		bottomRightCornerOfMatrix = _HHmatrix.subarray({{i+1, rows}, {i+1, cols}});  // {i, rows}
		NDArrayFactory<T>::mmulHelper(bottomRightCornerOfMatrix, &Prows, bottomRightCornerOfMatrix, (T)1., (T)0.);
		delete bottomRightCornerOfMatrix;
	}	
}






template class ND4J_EXPORT BiDiagonalUp<float>;
template class ND4J_EXPORT BiDiagonalUp<float16>;
template class ND4J_EXPORT BiDiagonalUp<double>;



}
}
}