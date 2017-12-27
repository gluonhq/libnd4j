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



///   Constructs a new singular value decomposition.
/// 
/// <param name="value"> The matrix to be decomposed.</param>
/// <param name="computeLeftSingularVectors"> Pass <see langword="true"/> if the left singular vector matrix U  should be computed. Pass <see langword="false"/> otherwise. Default is <see langword="true"/>.</param>
/// <param name="computeRightSingularVectors"> Pass <see langword="true"/> if the right singular vector matrix V  should be computed. Pass <see langword="false"/> otherwise. Default is <see langword="true"/>.</param>
/// <param name="autoTranspose"> Pass <see langword="true"/> to automatically transpose the value matrix in case JAMA's assumptions about the dimensionality of the matrix are violated. Pass <see langword="false"/> otherwise. Default is <see langword="false"/>.</param>
/// <param name="inPlace"> Pass <see langword="true"/> to perform the decomposition in place. The matrix <paramref name="value"/> will be destroyed in the process, resulting in less memory comsumption.</param>
/// 
template<typename T>
void svd(const NDArray<T>& matrix, NDArray<T>& u, NDArray<T>& s, NDArray<T>& v, const bool calcUV, const bool fullUV)    
public SingularValueDecomposition(Double[,] value, bool computeLeftSingularVectors, bool computeRightSingularVectors, bool autoTranspose, bool inPlace)
{

	int m = matrix.sizeAt(0);			// rows
	int n = matrix.sizeAt(1);			// Columns

	if(matrix.rankOf != 2 || m == 1 || n == 1)
		throw "ops::helpers::svd function: input array must be 2D matrix !";

	NDArray<T> a = matrix;				// use copy and leave input matrix unaffected
	bool swap = false;					// true when we transpose input matrix, that is when m < n

    if (m < n) {						// svd works with case m >= n, so we need to transpose input matrix and at the end  transpose swap the left and right eigenvectors

        a.transposei();
        // perform swapping
		m = m^n; n = n^m; m = m^n; 
        swap = true; 
    }


    int nu = System.Math.Min(m, n);
    int ni = System.Math.Min(m + 1, n);
    s = new Double[ni];
    u = new Double[m, nu];
    v = new Double[n, n];

    Double[] e = new Double[n];
    Double[] work = new Double[m];
    bool wantu = computeLeftSingularVectors;
    bool wantv = computeRightSingularVectors;

    // Will store ordered sequence of indices after sorting.
    si = new int[ni]; for (int i = 0; i < ni; i++) si[i] = i;


    // Reduce A to bidiagonal form, storing the diagonal elements in s and the super-diagonal elements in e.
    int nct = System.Math.Min(m - 1, n);
    int nrt = System.Math.Max(0, System.Math.Min(n - 2, m));
    int mrc = System.Math.Max(nct, nrt);

    for (int k = 0; k < mrc; k++)
    {
        if (k < nct)
        {
            // Compute the transformation for the k-th column and place the k-th diagonal in s[k].
            // Compute 2-norm of k-th column without under/overflow.
            s[k] = 0;
            for (int i = k; i < a.Rows(); i++)
                s[k] = Accord.Math.Tools.Hypotenuse(s[k], a[i, k]);

            if (s[k] != 0) 
            {
               if (a[k, k] < 0)
                  s[k] = -s[k];

               for (int i = k; i < a.Rows(); i++) 
                  a[i, k] /= s[k];
       
               a[k, k] += 1;
            }

            s[k] = -s[k];
        }

        for (int j = k+1; j < n; j++)
        {
            if ((k < nct) & (s[k] != 0))
            {
                // Apply the transformation.
                Double t = 0;
                for (int i = k; i < a.Rows(); i++)
                  t += a[i, k] * a[i, j];

               t = -t / a[k, k];

               for (int i = k; i < a.Rows(); i++)
                  a[i, j] += t * a[i, k];
             }

             // Place the k-th row of A into e for the
             // subsequent calculation of the row transformation.

             e[j] = a[k, j];
         }

         if (wantu & (k < nct))
         {
            // Place the transformation in U for subsequent back
            // multiplication.

            for (int i = k; i < a.Rows(); i++)
               u[i, k] = a[i, k];
         }

         if (k < nrt)
         {
            // Compute the k-th row transformation and place the
            // k-th super-diagonal in e[k].
            // Compute 2-norm without under/overflow.
            e[k] = 0;
            for (int i = k + 1; i < e.Rows(); i++)
               e[k] = Tools.Hypotenuse(e[k], e[i]);

            if (e[k] != 0)
            {
               if (e[k+1] < 0) 
                  e[k] = -e[k];

               for (int i = k + 1; i < e.Rows(); i++) 
                  e[i] /= e[k];

               e[k+1] += 1;
            }

            e[k] = -e[k];
            if ((k + 1 < m) & (e[k] != 0))
            {
                // Apply the transformation.
                for (int i = k + 1; i < work.Rows(); i++)
                    work[i] = 0;

                for (int i = k + 1; i < a.Rows(); i++)
                    for (int j = k + 1; j < a.Columns(); j++)
                        work[i] += e[j] * a[i, j];

               for (int j = k + 1; j < n; j++)
               {
                  Double t = -e[j] / e[k+1];
                  for (int i = k + 1; i < work.Rows(); i++) 
                     a[i, j] += t * work[i];
               }
            }

            if (wantv)
            {
                // Place the transformation in V for subsequent
                // back multiplication.

                for (int i = k + 1; i < v.Rows(); i++)
                   v[i, k] = e[i];
            }
        }
    }

    // Set up the final bidiagonal matrix or order p.
    int p = System.Math.Min(n, m + 1);
    if (nct < n) 
        s[nct] = a[nct, nct];
    if (m < p) 
        s[p - 1] = 0;
    if (nrt + 1 < p) 
        e[nrt] = a[nrt, p - 1];
    e[p - 1] = 0;

    // If required, generate U.
    if (wantu)
    {
        for (int j = nct; j < nu; j++)
        {
            for (int i = 0; i < u.Rows(); i++) 
                u[i, j] = 0;

            u[j, j] = 1;
        }

        for (int k = nct-1; k >= 0; k--)
        {
            if (s[k] != 0)
            {
                for (int j = k + 1; j < nu; j++)
                {
                    Double t = 0;
                    for (int i = k; i < u.Rows(); i++)
                        t += u[i, k] * u[i, j];

                    t = -t / u[k, k];

                    for (int i = k; i < u.Rows(); i++)
                        u[i, j] += t * u[i, k];
                }

                for (int i = k; i < u.Rows(); i++ )
                    u[i, k] = -u[i, k];

                u[k, k] = 1 + u[k, k];
                for (int i = 0; i < k - 1; i++) 
                    u[i, k] = 0;
            }
            else
            {
                for (int i = 0; i < u.Rows(); i++) 
                    u[i, k] = 0;
                u[k, k] = 1;
            }
            }
    }
      

    // If required, generate V.
    if (wantv)
    {
        for (int k = n - 1; k >= 0; k--)
        {
            if ((k < nrt) & (e[k] != 0))
            {
                // TODO: The following is a pseudo correction to make SVD
                //  work on matrices with n > m (less rows than columns).

                // For the proper correction, compute the decomposition of the
                //  transpose of A and swap the left and right eigenvectors

                // Original line:
                //   for (int j = k + 1; j < nu; j++)
                // Pseudo correction:
                //   for (int j = k + 1; j < n; j++)

                for (int j = k + 1; j < n; j++) // pseudo-correction
                {
                    Double t = 0;
                    for (int i = k + 1; i < v.Rows(); i++)
                        t += v[i, k] * v[i, j];

                    t = -t / v[k+1, k];
                    for (int i = k + 1; i < v.Rows(); i++)
                        v[i, j] += t * v[i, k];
                }
            }

            for (int i = 0; i < v.Rows(); i++)
                v[i, k] = 0;
            v[k, k] = 1;
        }
    }

    // Main iteration loop for the singular values.

    int pp = p-1;
    int iter = 0;
    Double eps = Constants.DoubleEpsilon;
    while (p > 0)
    {
        int k,kase;

        // Here is where a test for too many iterations would go.

        // This section of the program inspects for
        // negligible elements in the s and e arrays.  On
        // completion the variables kase and k are set as follows.

        // kase = 1     if s(p) and e[k-1] are negligible and k<p
        // kase = 2     if s(k) is negligible and k<p
        // kase = 3     if e[k-1] is negligible, k<p, and
        //              s(k), ..., s(p) are not negligible (qr step).
        // kase = 4     if e(p-1) is negligible (convergence).

        for (k = p - 2; k >= -1; k--)
        {
            if (k == -1)
                break;

            var alpha = tiny + eps * (System.Math.Abs(s[k]) + System.Math.Abs(s[k + 1]));
            if (System.Math.Abs(e[k]) <= alpha || Double.IsNaN(e[k]))
            {
                e[k] = 0;
                break;
            }
        }

        if (k == p-2)
            kase = 4;

        else
        {
            int ks;
            for (ks = p - 1; ks >= k; ks--)
            {
               if (ks == k)
                  break;

               Double t = (ks != p     ? Math.Abs(e[ks])   : (Double)0) + 
                          (ks != k + 1 ? Math.Abs(e[ks-1]) : (Double)0);

               if (Math.Abs(s[ks]) <= eps*t) 
               {
                  s[ks] = 0;
                  break;
               }
            }

            if (ks == k)
               kase = 3;

            else if (ks == p-1)
               kase = 1;

            else
            {
               kase = 2;
               k = ks;
            }
         }

         k++;

         // Perform the task indicated by kase.
         switch (kase)
         {
            // Deflate negligible s(p).
            case 1:
            {
               Double f = e[p - 2];
               e[p-2] = 0;
               for (int j = p - 2; j >= k; j--) 
               {
                  Double t = Tools.Hypotenuse(s[j],f);
                  Double cs = s[j] / t;
                  Double sn = f / t;
                  s[j] = t;
                  if (j != k) 
                  {
                     f = -sn * e[j - 1];
                     e[j - 1] = cs * e[j - 1];
                  }
                  if (wantv) 
                  {
                     for (int i = 0; i < v.Rows(); i++) 
                     {
                        t = cs * v[i, j] + sn * v[i, p-1];
                        v[i, p-1] = -sn * v[i, j] + cs * v[i, p-1];
                        v[i, j] = t;
                     }
                  }
               }
            }
            break;

            // Split at negligible s(k).

            case 2:
            {
               Double f = e[k - 1];
               e[k - 1] = 0;
               for (int j = k; j < p; j++)
               {
                  Double t = Tools.Hypotenuse(s[j], f);
                  Double cs = s[j] / t;
                  Double sn = f / t;
                  s[j] = t;
                  f = -sn * e[j];
                  e[j] = cs * e[j];
                  if (wantu) 
                  {
                        for (int i = 0; i < u.Rows(); i++) 
                        {
                            t = cs * u[i, j] + sn * u[i, k-1];
                            u[i, k - 1] = -sn * u[i, j] + cs * u[i, k-1];
                            u[i, j] = t;
                        }
                  }
               }
            }
            break;

            // Perform one qr step.
            case 3:
                {
                   // Calculate the shift.
                   Double scale = Math.Max(Math.Max(Math.Max(Math.Max(
                           Math.Abs(s[p-1]),Math.Abs(s[p-2])),Math.Abs(e[p-2])), 
                           Math.Abs(s[k])),Math.Abs(e[k]));
                   Double sp = s[p-1] / scale;
                   Double spm1 = s[p-2] / scale;
                   Double epm1 = e[p-2] / scale;
                   Double sk = s[k] / scale;
                   Double ek = e[k] / scale;
                   Double b = ((spm1 + sp)*(spm1 - sp) + epm1*epm1)/2;
                   Double c = (sp*epm1)*(sp*epm1);
                   Double shift = 0;
                   if ((b != 0) || (c != 0))
                   {
                    if (b < 0)
                        shift = -(Double)System.Math.Sqrt(b * b + c);
                    else
                        shift = (Double)System.Math.Sqrt(b * b + c);
                      shift = c / (b + shift);
                   }

                   Double f = (sk + sp)*(sk - sp) + (Double)shift;
                   Double g = sk*ek;

                   // Chase zeros.
                   for (int j = k; j < p - 1; j++)
                   {
                      Double t = Tools.Hypotenuse(f, g);
                      Double cs = f / t;
                      Double sn = g / t;

                      if (j != k)
                         e[j - 1] = t;

                      f = cs * s[j] + sn * e[j];
                      e[j] = cs * e[j] - sn * s[j];
                      g = sn * s[j + 1];
                      s[j+1] = cs * s[j + 1];

                      if (wantv)
                      {
                         for (int i = 0; i < v.Rows(); i++)
                         {
                            t = cs * v[i, j] + sn * v[i, j + 1];
                            v[i, j + 1] = -sn*v[i, j] + cs*v[i, j + 1];
                            v[i, j] = t;
                         }
                      }

                      t = Tools.Hypotenuse(f,g);
                      cs = f / t;
                      sn = g / t;
                      s[j] = t;
                      f = cs * e[j] + sn * s[j + 1];
                      s[j + 1] = -sn * e[j] + cs * s[j + 1];
                      g = sn * e[j + 1];
                      e[j + 1] = cs * e[j + 1];

                      if (wantu && (j < m - 1))
                      {
                         for (int i = 0; i < u.Rows(); i++)
                         {
                            t = cs * u[i, j] + sn * u[i, j + 1];
                            u[i, j + 1] = -sn * u[i, j] + cs * u[i, j + 1];
                            u[i, j] = t;
                         }
                      }
                   }

                   e[p - 2] = f;
                   iter = iter + 1;
                }
                break;

            // Convergence.
            case 4:
                {
                    // Make the singular values positive.
                    if (s[k] <= 0)
                    {
                        s[k] = (s[k] < 0 ? -s[k] : (Double)0);

                        if (wantv)
                        {
                            for (int i = 0; i <= pp; i++) 
                                v[i, k] = -v[i, k];
                        }
                    }

                    // Order the singular values.
                    while (k < pp)
                    {
                        if (s[k] >= s[k + 1])
                            break;

                        Double t = s[k];
                        s[k] = s[k + 1];
                        s[k+1] = t;
                        if (wantv && (k < n - 1))
                        {
                            for (int i = 0; i < n; i++)
                            {
                                t = v[i, k + 1];
                                v[i, k + 1] = v[i, k]; 
                                v[i, k] = t;
                            }
                        }

                        if (wantu && (k < m - 1))
                        {
                            for (int i = 0; i < u.Rows(); i++)
                            {
                                t = u[i, k + 1]; 
                                u[i, k + 1] = u[i, k]; 
                                u[i, k] = t;
                            }
                        }

                        k++;
                    }

                    iter = 0;
                    p--;
                }
                break;
        }
    }
    

    // If we are violating JAMA's assumption about 
    // the input dimension, we need to swap u and v.
    if (swapped)
    {
        var temp = this.u;
        this.u = this.v;
        this.v = temp;
    }
}




















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