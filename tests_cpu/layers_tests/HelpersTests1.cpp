#include "testlayers.h"
#include <ops/declarable/helpers/householder.h>
#include <ops/declarable/helpers/biDiagonalUp.h>


using namespace nd4j;

class HelpersTests1 : public testing::Test {
public:
    
    HelpersTests1() {
        
        std::cout<<std::endl<<std::flush;
    }

};


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHHmatrix_test1) {
            
    NDArray<double> x('c', {1,9}, {14,17,3,1,9,1,2,5,11});                
    double coeffExpected = 1.51923;
    double normXExpected = -26.96294;

    double coeff, normX;    
    ops::helpers::Householder<double>::evalHHmatrix(x, coeff, normX);
    // tail.printBuffer();

    ASSERT_NEAR(normX, normXExpected, 1e-5);
    ASSERT_NEAR(coeff, coeffExpected, 1e-5);    

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHHmatrix_test2) {
            
    NDArray<double> x('c', {1,4}, {14,17,3,1});            
    NDArray<double> pExp('c', {4,4}, {-0.629253, -0.764093,   -0.13484, -0.0449467, -0.764093,  0.641653, -0.0632377, -0.0210792, -0.13484,-0.0632377,    0.98884,-0.00371987, -0.0449467,-0.0210792,-0.00371987,  0.99876});

    double normX, coeff;
    NDArray<double> result = ops::helpers::Householder<double>::evalHHmatrix(x, coeff, normX);

    ASSERT_TRUE(result.isSameShapeStrict(&pExp));
    ASSERT_TRUE(result.equalsTo(&pExp));

}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, BiDiagonalizeUp_test1) {
            
    NDArray<double> matrix('c', {4,4}, {9,13,3,6,13,11,7,6,3,7,4,7,6,6,7,10});            
    NDArray<double> hhMatrixExp('c', {4,4}, {1.524,  1.75682,0.233741,0.289458, 0.496646,   1.5655, 1.02929,0.971124, 0.114611,-0.451039, 1.06367, 0, 0.229221,-0.272237,0.938237,0,});
    
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    object._HHmatrix.printBuffer();

    ASSERT_TRUE(hhMatrixExp.isSameShapeStrict(&object._HHmatrix));
    ASSERT_TRUE(hhMatrixExp.equalsTo(&object._HHmatrix));
}
    