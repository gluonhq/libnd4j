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
		
	NDArray<T>* bottomRightCorner(nullptr), *column(nullptr), *row(nullptr);
	T coeff, normX;
	for(int i = 0; i < cols-1; ++i ) {

		// evaluate Householder matrix nullifying columns 		
		column = _HHmatrix.subarray({{i, rows}, {i, i+1}});						
		NDArray<T> Pcols = Householder<T>::evalHHmatrix(*column, coeff, normX);		
		_HHmatrix(i,i) = coeff;
		_HHbidiag(i,i) = normX;		
		// multiply given matrix block on householder matrix from the left: Pcols * bottomRightCorner		
		bottomRightCorner =  _HHmatrix.subarray({{i, rows}, {i+1, cols}});	// {i, cols}				
		NDArray<T> temp1 = *bottomRightCorner;
		
		if(!Pcols.isScalar()) {
			
			NDArrayFactory<T>::mmulHelper(&Pcols, &temp1, &temp1, (T)1., (T)0.);								
			bottomRightCorner->assign(&temp1);
		}
		else
			*bottomRightCorner *= Pcols(0);			
		
		if(i == cols-2) {
			
			_HHbidiag(i,i+1)   = (*bottomRightCorner)(0);
			_HHbidiag(i+1,i+1) = (*bottomRightCorner)(1);			
			_HHmatrix(i,i+1)   = (T).0;
			_HHmatrix(i+1,i+1) = (T).0;						
			delete bottomRightCorner;
			delete column;
			continue; 							// do not apply right multiplying at last iteration
		}
		
		delete bottomRightCorner;
		delete column;
		
		// evaluate Householder matrix nullifying rows 
		row = _HHmatrix.subarray({{i, i+1}, {i+1, cols}});
		NDArray<T> Prows = Householder<T>::evalHHmatrix(*row, coeff, normX);		
		_HHmatrix(i,i+1) = coeff;
		_HHbidiag(i,i+1) = normX;
		// multiply given matrix block on householder matrix from the right: bottomRightCorner * Prols
		bottomRightCorner = _HHmatrix.subarray({{i+1, rows}, {i+1, cols}});  // {i, rows}
		NDArray<T> temp2 = *bottomRightCorner;
		
		if(!Prows.isScalar()) {
			
			NDArrayFactory<T>::mmulHelper(&temp2, &Prows, &temp2, (T)1., (T)0.);
			bottomRightCorner->assign(&temp2);
		}
		else
			*bottomRightCorner *= Prows(0);			
		
		delete bottomRightCorner;
		delete row;
	}	
}






template class ND4J_EXPORT BiDiagonalUp<float>;
template class ND4J_EXPORT BiDiagonalUp<float16>;
template class ND4J_EXPORT BiDiagonalUp<double>;



}
}
}