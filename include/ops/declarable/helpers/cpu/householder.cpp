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
NDArray<T> evalHouseholderMatrix(const NDArray<T>& x, T& normX) {

	// input validation
	if(!x.isVector() && !x.isScalar())
		throw "ops::helpers::evalHouseholderMatrix function: input array must be vector or scalar!";
			
	NDArray<T> w((int)x.lengthOf(), 1,  x.ordering(), x.getWorkspace());							// column-vector
	NDArray<T> wT(1, (int)x.lengthOf(), x.ordering(), x.getWorkspace());							// row-vector (transposed w)	

	normX = x.template reduceNumber<simdOps::Norm2<T>>();	
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
		throw "ops::helpers::biDiagonalizeUp function: this procedure is applicable only for input matrix with rows >= cols !";
		
	T normX;
	NDArray<T>* bottomRightCornerOfMatrix = nullptr;
	for(int i = 0; i < cols-1; ++i ) {

		// evaluate Householder matrix which nullifies columns 
		NDArray<T> Pcols = evalHouseholderMatrix(matrix({{i, rows}, {i, i+1}}), normX);
		// multiply given matrix block on householder matrix from the left: Pcols * bottomRightCornerOfMatrix
		bottomRightCornerOfMatrix =  matrix.subarray({{i, rows}, {i, cols}});	// {i+1, cols}
		NDArrayFactory<T>::mmulHelper(&Pcols, bottomRightCornerOfMatrix, bottomRightCornerOfMatrix, (T)1., (T)0.);
		delete bottomRightCornerOfMatrix;

		if(i == cols-2) continue; 							// do not apply right multiplying on last iteration

		// evaluate Householder matrix which nullifies rows 
		NDArray<T> Prows = evalHouseholderMatrix(matrix({{i, i+1}, {i+1, cols}}), normX);		
		// multiply given matrix block on householder matrix from the right: bottomRightCornerOfMatrix * Prols
		bottomRightCornerOfMatrix = matrix.subarray({{i, rows}, {i+1, cols}});  // {i+1, rows}
		NDArrayFactory<T>::mmulHelper(bottomRightCornerOfMatrix, &Prows, bottomRightCornerOfMatrix, (T)1., (T)0.);
		delete bottomRightCornerOfMatrix;
	}

	
}


template<typename T>
void svd(NDArray<T>& u, NDArray<T>& s, NDArray<T>& v, const bool calcUV = false, const bool fullUV = false)
{

  const RealScalar considerZero = (std::numeric_limits<RealScalar>::min)();
  
 
  //**** step 0 - Copy the input matrix and apply scaling to reduce over/under-flows
  RealScalar scale = matrix.cwiseAbs().maxCoeff();
  if(scale==Literal(0)) scale = Literal(1);
  MatrixX copy;
  if (m_isTranspose) copy = matrix.adjoint()/scale;
  else               copy = matrix/scale;
  
  //**** step 1 - Bidiagonalization
  // FIXME this line involves temporaries
  internal::UpperBidiagonalization<MatrixX> bid(copy);

  //**** step 2 - Divide & Conquer
  m_naiveU.setZero();
  m_naiveV.setZero();
  // FIXME this line involves a temporary matrix
  m_computed.topRows(m_diagSize) = bid.bidiagonal().toDenseMatrix().transpose();
  m_computed.template bottomRows<1>().setZero();
  divide(0, m_diagSize - 1, 0, 0, 0);

  //**** step 3 - Copy singular values and vectors
  for (int i=0; i<m_diagSize; i++)
  {
    RealScalar a = abs(m_computed.coeff(i, i));
    m_singularValues.coeffRef(i) = a * scale;
    if (a<considerZero)
    {
      m_nonzeroSingularValues = i;
      m_singularValues.tail(m_diagSize - i - 1).setZero();
      break;
    }
    else if (i == m_diagSize - 1)
    {
      m_nonzeroSingularValues = i + 1;
      break;
    }
  }


  if(m_isTranspose) copyUV(bid.householderV(), bid.householderU(), m_naiveV, m_naiveU);
  else              copyUV(bid.householderU(), bid.householderV(), m_naiveU, m_naiveV);

  m_isInitialized = true;
  return *this;
}// end compute

template void evalHouseholderData<float>  (const NDArray<float  >& x, NDArray<float  >& tail, float  & normX, float  & coeff);
template void evalHouseholderData<float16>(const NDArray<float16>& x, NDArray<float16>& tail, float16& normX, float16& coeff);
template void evalHouseholderData<double> (const NDArray<double >& x, NDArray<double >& tail, double & normX, double & coeff);

template NDArray<float>   evalHouseholderMatrix<float>  (const NDArray<float  >& x, float  & normX);
template NDArray<float16> evalHouseholderMatrix<float16>(const NDArray<float16>& x, float16& normX);
template NDArray<double>  evalHouseholderMatrix<double> (const NDArray<double >& x, double & normX);

template void biDiagonalizeUp<float>  (NDArray<float  >& matrix);
template void biDiagonalizeUp<float16>(NDArray<float16>& matrix);
template void biDiagonalizeUp<double> (NDArray<double >& matrix);

}
}
}