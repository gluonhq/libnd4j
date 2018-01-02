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
            
    NDArray<double> x('c', {1,4}, {14,17,3,1});                
    NDArray<double> exp('c', {4,4}, {-0.629253, -0.764093,   -0.13484, -0.0449467, -0.764093,  0.641653, -0.0632377, -0.0210792, -0.13484,-0.0632377,    0.98884,-0.00371987, -0.0449467,-0.0210792,-0.00371987,    0.99876});
    
    NDArray<double> result = ops::helpers::Householder<double>::evalHHmatrix(x);    

    ASSERT_TRUE(result.isSameShapeStrict(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHHmatrix_test2) {
            
    NDArray<double> x('c', {1,3}, {14,-4,3});                
    NDArray<double> exp('c', {3,3}, {-0.941742, 0.269069,-0.201802, 0.269069, 0.962715,0.0279639, -0.201802,0.0279639, 0.979027});
    
    NDArray<double> result = ops::helpers::Householder<double>::evalHHmatrix(x);    

    ASSERT_TRUE(result.isSameShapeStrict(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));

}


/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHHmatrixData_test1) {
            
    NDArray<double> x('c', {1,4}, {14,17,3,1});            
    NDArray<double> tail('c', {1,3});            
    NDArray<double> expTail('c', {1,3}, {0.468984, 0.0827618, 0.0275873});
    const double normXExpected = -22.2486;
    const double coeffExpected = 1.62925;

    double normX, coeff;
    ops::helpers::Householder<double>::evalHHmatrixData(x, tail, coeff, normX);
    // expTail.printShapeInfo();

    ASSERT_NEAR(normX, normXExpected, 1e-5);
    ASSERT_NEAR(coeff, coeffExpected, 1e-5);
    ASSERT_TRUE(tail.isSameShapeStrict(&expTail));
    ASSERT_TRUE(tail.equalsTo(&expTail));

}


/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, Householder_mulLeft_test1) {
            
    NDArray<double> x('c', {4,4}, {12 ,19 ,14 ,3 ,10 ,4 ,17 ,19 ,19 ,18 ,5 ,3 ,6 ,4 ,2 ,16});            
    NDArray<double> tail('c', {1,3}, {0.5,0.5,0.5});            
    NDArray<double> exp('c', {4,4}, {9.05,15.8,11.4, 0.8, 8.525, 2.4,15.7,17.9, 17.525,16.4, 3.7, 1.9, 4.525, 2.4, 0.7,14.9});
    
    ops::helpers::Householder<double>::mulLeft(x, tail, 0.1);
    // expTail.printShapeInfo();

    ASSERT_TRUE(x.isSameShapeStrict(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));

}

/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, Householder_mulLeft_test2) {
            
    NDArray<double> x('c', {4,4}, {12 ,19 ,14 ,3 ,10 ,4 ,17 ,19 ,19 ,18 ,5 ,3 ,6 ,4 ,2 ,16});            
    NDArray<double> tail('c', {3,1}, {0.5,0.5,0.5});            
    NDArray<double> exp('c', {4,4}, {9.05,15.8,11.4, 0.8, 8.525, 2.4,15.7,17.9, 17.525,16.4, 3.7, 1.9, 4.525, 2.4, 0.7,14.9});
    
    ops::helpers::Householder<double>::mulLeft(x, tail, 0.1);    

    ASSERT_TRUE(x.isSameShapeStrict(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));

}

/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, Householder_mulRight_test1) {
            
    NDArray<double> x('c', {4,4}, {12 ,19 ,14 ,3 ,10 ,4 ,17 ,19 ,19 ,18 ,5 ,3 ,6 ,4 ,2 ,16});            
    NDArray<double> tail('c', {1,3}, {0.5,0.5,0.5});            
    NDArray<double> exp('c', {4,4}, {9,17.5,12.5,  1.5, 7, 2.5,15.5, 17.5, 15.8,16.4, 3.4,  1.4, 4.3,3.15,1.15,15.15});
    
    ops::helpers::Householder<double>::mulRight(x, tail, 0.1);    

    ASSERT_TRUE(x.isSameShapeStrict(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));

}


/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, BiDiagonalizeUp_test1) {
            
    NDArray<double> matrix('c', {4,4}, {9,13,3,6,13,11,7,6,3,7,4,7,6,6,7,10});      
    NDArray<double> hhMatrixExp('c', {4,4}, {1.524000,  1.75682,0.233741,0.289458, 0.496646,   1.5655, 1.02929,0.971124, 0.114611,-0.451039, 1.06367,0, 0.229221,-0.272237,0.938237,0});
    NDArray<double> hhBidiagExp('c', {4,4}, {-17.1756, 24.3869,       0,      0, 0,-8.61985,-3.89823,      0, 0,       0, 4.03047,4.13018, 0,       0,       0,1.21666});
    
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    object._HHmatrix.printBuffer();
    object._HHbidiag.printBuffer();

    ASSERT_TRUE(hhMatrixExp.isSameShapeStrict(&object._HHmatrix));
    ASSERT_TRUE(hhMatrixExp.equalsTo(&object._HHmatrix));
    ASSERT_TRUE(hhBidiagExp.isSameShapeStrict(&object._HHbidiag));
    ASSERT_TRUE(hhBidiagExp.equalsTo(&object._HHbidiag));
}
    
///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, BiDiagonalizeUp_test2) {
            
    NDArray<double> matrix('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});      
    NDArray<double> hhMatrixExp('c', {5,4}, {1.52048, 1.37012, 0.636326, -0.23412, 0.494454, 1.66025,  1.66979,-0.444696, 0.114105,0.130601, 1.58392,        0, -0.22821, 0.215638,0.0524781,  1.99303, 0.0760699,0.375605, 0.509835,0.0591568});
    NDArray<double> hhBidiagExp('c', {4,4}, {-17.2916,7.03123,       0,       0, 0, 16.145,-22.9275,       0, 0,      0, -9.9264,-11.5516, 0,      0,       0,-12.8554});
    
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    object._HHmatrix.printBuffer();
    object._HHbidiag.printBuffer();

    ASSERT_TRUE(hhMatrixExp.isSameShapeStrict(&object._HHmatrix));
    ASSERT_TRUE(hhMatrixExp.equalsTo(&object._HHmatrix));
    ASSERT_TRUE(hhBidiagExp.isSameShapeStrict(&object._HHbidiag));
    ASSERT_TRUE(hhBidiagExp.equalsTo(&object._HHbidiag));
}
    
///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, BiDiagonalizeUp_test3) {
            
    NDArray<double> matrix('c', {6,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12, 0,-15,10,2});      
    NDArray<double> hhMatrixExp('c', {6,4}, {1.52048,  1.37012, 0.636326, -0.23412, 0.494454,  1.65232,  1.59666,-0.502606, 0.114105, 0.129651,  1.35075,        0, -0.22821, 0.214071, 0.103749,  1.61136, 0.0760699, 0.372875, 0.389936,   0.2398, 0,0.0935171,-0.563777, 0.428587});
    NDArray<double> hhBidiagExp('c', {4,4}, {-17.2916,7.03123,       0,      0, 0,16.3413,-20.7828,      0, 0,      0,-18.4892,4.13261, 0,      0,       0,-21.323});
    
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    object._HHmatrix.printBuffer();
    object._HHbidiag.printBuffer();

    ASSERT_TRUE(hhMatrixExp.isSameShapeStrict(&object._HHmatrix));
    ASSERT_TRUE(hhMatrixExp.equalsTo(&object._HHmatrix));
    ASSERT_TRUE(hhBidiagExp.isSameShapeStrict(&object._HHbidiag));
    ASSERT_TRUE(hhBidiagExp.equalsTo(&object._HHbidiag));
}
    