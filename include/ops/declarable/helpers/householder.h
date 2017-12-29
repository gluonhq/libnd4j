//
// Created by Yurii Shyrma on 18.12.2017.
//

#ifndef LIBND4J_HOUSEHOLDER_H
#define LIBND4J_HOUSEHOLDER_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {


template<typename T>
class Householder {

    public:
        
    /**
    *  this function calculates Householder matrix P = identity_matrix - coeff * w * w^T
    *  P * x = [normX, 0, 0 , 0, ...]
    *  coeff - scalar    
    *  w = [1, w1, w2, w3, ...]
    *  w = u / u0
    *  u = x - |x|*e0
    *  u0 = x0 - |x| 
    *  e0 = [1, 0, 0 , 0, ...]
    * 
    *  x - input vector, remains unaffected
    *  normX - this scalar is the first non-zero element in vector resulting from Householder transformation -> (P*x)
    *  coeff - scalar, scaling factor in Householder matrix formula  
    */                       
    static NDArray<T> evalHHmatrix(NDArray<T>& x, T& coeff, T& normX);
        


};
    // /**
    // *  this function evaluates data (coeff, normX, tail) used in Householder transformation
    // *  formula for Householder matrix: P = identity_matrix - coeff * w * w^T
    // *  P * x = [normX, 0, 0 , 0, ...]
    // *  coeff - scalar    
    // *  w = [1, w1, w2, w3, ...], "tail" is w except first unity element, that is "tail" = [w1, w2, w3, ...]
    // *  w = u / u0
    // *  u = x - |x|*e0
    // *  u0 = x0 - |x| 
    // *  e0 = [1, 0, 0 , 0, ...]
    // * 
    // *  x - input vector, remains unaffected
    // *  tail - output vector with length = x.lengthOf() - 1 and contains all elements of w vector except first one 
    // *  normX - this scalar is the first non-zero element in vector resulting from Householder transformation -> (P*x)  
    // *  coeff - scalar, scaling factor in Householder matrix formula  
    // */                  	
    // template <typename T>
    // void evalHouseholderData(const NDArray<T>& x, NDArray<T>& tail, T& normX, T& coeff);


    // /**
    // *  this function calculates Householder matrix P = identity_matrix - coeff * w * w^T
    // *  P * x = [normX, 0, 0 , 0, ...]
    // *  coeff - scalar    
    // *  w = [1, w1, w2, w3, ...]
    // *  w = u / u0
    // *  u = x - |x|*e0
    // *  u0 = x0 - |x| 
    // *  e0 = [1, 0, 0 , 0, ...]
    // * 
    // *  x - input vector, remains unaffected
    // *  normX - this scalar is the first non-zero element in vector resulting from Householder transformation -> (P*x)
    // */                  	
    // template <typename T>
    // NDArray<T> evalHHmatrix(const NDArray<T>& x, T& normX);

    
    // /**
    // *  this function reduce given matrix to  upper bidiagonal form (in-place operation), matrix must satisfy following condition rows >= cols
    // * 
    // *  matrix - input 2D matrix to be reduced to upper bidiagonal from    
    // */
    // template <typename T>
    // void biDiagonalizeUp(NDArray<T>& matrix);

    // /** 
    // *  given a matrix matrix [m,n], this function computes its singular value decomposition matrix = u * s * v^T
    // *   
    // *  matrix - input 2D matrix to decompose, [m, n]
    // *  u - unitary matrix containing left singular vectors of input matrix, [m, m]
    // *  s - diagonal matrix with singular values of input matrix (non-negative) on the diagonal sorted in decreasing order,
    // *      actually the mathematically correct dimension of s is [m, n], however for memory saving we work with s as vector [1, p], where p is smaller among m and n
    // *  v - unitary matrix containing right singular vectors of input matrix, [n, n]
    // *  calcUV - if true then u and v will be computed, in opposite case function works significantly faster
    // *  fullUV - if false then only p (p is smaller among m and n) first columns of u and v will be calculated and their dimensions in this case are [m, p] and [n, p]
    // *
    // */
    // void svd(const NDArray<T>& matrix, NDArray<T>& u, NDArray<T>& s, NDArray<T>& v, const bool calcUV = false, const bool fullUV = false)    



}
}
}


#endif //LIBND4J_HOUSEHOLDER_H
