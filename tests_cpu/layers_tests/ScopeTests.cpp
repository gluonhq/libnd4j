//
// Created by raver119 on 15.10.2017.
//

#include "testlayers.h"
#include <Graph.h>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::graph;

class ScopeTests : public testing::Test {
public:

};

TEST_F(ScopeTests, BasicTests_1) {
    Graph<float> graph;

    auto x = new NDArray<float>('c', {2, 2});
    x->assign(0.0f);

    auto variableSpace = graph.getVariableSpace();
    variableSpace->putVariable(-1, x);

    nd4j::ops::Scope<float> opScope;

    auto scopeBody = new Node<float>(OpType_LOGIC, 10, 1);
    scopeBody->setName("scopeBody");
    scopeBody->setCustomOp(&opScope);


    graph.addNode(scopeBody);

    ASSERT_EQ(1, graph.totalNodes());

    auto scopedB0 = new Node<float>(OpType_SCALAR, 0, 6, {-1}, {}, {}, 1.0f);
    scopedB0->markInplace(true);
    scopedB0->setScopeInfo(1, "scopeBody");

    graph.addNode(scopedB0);

    ASSERT_EQ(1, graph.totalNodes());

}

TEST_F(ScopeTests, RealTests_1) {
    Graph<float> graph;

    auto x = new NDArray<float>('c', {2, 2});
    x->assign(0.0f);

    auto y = new NDArray<float>('c', {2, 2});
    y->assign(0.0);

    auto scalar = new NDArray<float>('c', {1, 1});
    scalar->putScalar(0, 10);

    auto variableSpace = graph.getVariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    variableSpace->putVariable(-3, scalar);

    // just few ops coming before while
    auto nodeA = new Node<float>(OpType_TRANSFORM, 35, 1, {-1});
    auto nodeB = new Node<float>(OpType_SCALAR, 0, 2, {1}, {}, {}, 1.0);

    //
    auto scopeCondition = new Node<float>(OpType_LOGIC, 10, 3);
    scopeCondition->setName("scopeCondition");
    nd4j::ops::Scope<float> opScope;
    scopeCondition->setCustomOp(&opScope);

    // this is scope of the body, it'll be executed multiple times
    auto scopeBody = new Node<float>(OpType_LOGIC, 10, 10);
    scopeBody->setName("scopeBody");
    scopeBody->setCustomOp(&opScope);

////////////////////////////////////////////////////////////////////////////////////////////////////
//// filling out condition scope
////////////////////////////////////////////////////////////////////////////////////////////////////
    // this is Sum accumulation, which feed
    auto scopedA0 = new Node<float>(OpType_ACCUMULATION, 0, 4, {12});
    scopedA0->setScopeInfo(3, "scopeCondition");

    // this op compares LT A0 result with variable `scalar` which is 10;
    auto scopedA1 = new Node<float>(OpType_BOOLEAN, 0, 5, {4, -3});
    nd4j::ops::lt_scalar<float> op;
    scopedA1->setCustomOp(&op);
    scopedA1->setScopeInfo(3, "scopeCondition");


////////////////////////////////////////////////////////////////////////////////////////////////////
//// filling out body scope
////////////////////////////////////////////////////////////////////////////////////////////////////
    auto scopedB0 = new Node<float>(OpType_SCALAR, 0, 6, {12}, {}, {}, 1.0f);
    scopedB0->markInplace(false);
    scopedB0->setScopeInfo(10, "scopeBody");

    auto nodeReturn = new Node<float>(OpType_LOGIC, 40, 7, {6}, {12});
    nd4j::ops::Return<float> opReturn;
    nodeReturn->setCustomOp(&opReturn);
    nodeReturn->setScopeInfo(10, "scopeBody");

    // WHILE operations takes 2 scopes - :0 is condition scope, and :1 is loop body scope
    auto nodeWhile = new Node<float>(OpType_LOGIC, 0, 12, {-2, 3, 10});
    nd4j::ops::While<float> opWhile;
    nodeWhile->setCustomOp(&opWhile);

    // adding root nodes first, nothing unusual expected here
    graph.addNode(nodeA);
    graph.addNode(nodeB);

    // now we're registering our scopes
    graph.addNode(scopeCondition);
    graph.addNode(scopeBody);

    // at this moment graph should have 4 (four) nodes registered
    ASSERT_EQ(4, graph.totalNodes());

    // adding node that's attached to some scope. so it should be pushed to specific scope
    graph.addNode(scopedA0);

    // we should still have 4 ops in graph, because node added above - goes directly into the scope
    // thus falls out of the graph direct execution - it can be executed only via Scope
    ASSERT_EQ(4, graph.totalNodes());

    graph.addNode(scopedA1);
    graph.addNode(scopedB0);
    graph.addNode(nodeReturn);

    // should be still 4. no options here.
    ASSERT_EQ(4, graph.totalNodes());

    // WHILE is valid node, so we expect nodes counter to go up
    graph.addNode(nodeWhile);
    ASSERT_EQ(5, graph.totalNodes());

    // now, let's try to execute graph
    Nd4jStatus status = GraphExecutioner<float>::execute(&graph);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto w = variableSpace->getVariable(12, 0)->getNDArray();

    ASSERT_NEAR(40.f, w->sumNumber(), 1e-5f);
}