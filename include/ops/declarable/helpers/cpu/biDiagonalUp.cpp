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
	_HHbidiag.assign(0.);
	
	evalData();

}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void BiDiagonalUp<T>::evalData() {
	
	const int rows = _HHmatrix.sizeAt(0);
	const int cols = _HHmatrix.sizeAt(1);
	
	if(rows < cols)
		throw "ops::helpers::BiDiagonalizeUp::evalData method: this procedure is applicable only for input matrix with rows >= cols !";
		
	NDArray<T>* bottomRightCorner(nullptr), *column(nullptr), *row(nullptr), *tail(nullptr);	
	T coeff, normX;
	for(int i = 0; i < cols-1; ++i ) {

		// evaluate Householder matrix nullifying columns 		
		column = _HHmatrix.subarray({{i,   rows}, {i, i+1}});						
		tail   = _HHmatrix.subarray({{i+1, rows}, {i, i+1}});
		Householder<T>::evalHHmatrixData(*column, *tail, _HHmatrix(i,i), _HHbidiag(i,i)); 
		// multiply corresponding matrix block on householder matrix from the left: P * bottomRightCorner		
		bottomRightCorner =  _HHmatrix.subarray({{i, rows}, {i+1, cols}});	// {i, cols}				
		Householder<T>::mulLeft(*bottomRightCorner, *tail, _HHmatrix(i,i));

		delete bottomRightCorner;
		delete column;
		delete tail;

		
		if(i == cols-2) {
			
		// 	_HHbidiag(i,i+1)   = (*bottomRightCorner)(0);
		// 	_HHbidiag(i+1,i+1) = (*bottomRightCorner)(1);			
		// 	_HHmatrix(i,i+1)   = (T).0;
		// 	_HHmatrix(i+1,i+1) = (T).0;						
		// 	delete bottomRightCorner;
		// 	delete column;
		// 	delete tail;
			continue; 							// do not apply right multiplying at last iteration
		}
		
		
		// evaluate Householder matrix nullifying rows 
		row  = _HHmatrix.subarray({{i, i+1}, {i+1, cols}});
		tail = _HHmatrix.subarray({{i, i+1}, {i+2, cols}});
		Householder<T>::evalHHmatrixData(*row, *tail, _HHmatrix(i,i+1), _HHbidiag(i,i+1));
		// multiply corresponding matrix block on householder matrix from the right: bottomRightCorner * P
		bottomRightCorner = _HHmatrix.subarray({{i+1, rows}, {i+1, cols}});  // {i, rows}
		Householder<T>::mulRight(*bottomRightCorner, *tail, _HHmatrix(i,i+1));
				
		
		delete bottomRightCorner;
		delete row;
		delete tail;
	}	
}






template class ND4J_EXPORT BiDiagonalUp<float>;
template class ND4J_EXPORT BiDiagonalUp<float16>;
template class ND4J_EXPORT BiDiagonalUp<double>;



}
}
}