//
// Created by raver119 on 17.10.2017.
//

#include "ops/declarable/LegacyStatsOp.h"

namespace nd4j {
    namespace ops {
        template <typename T>
        Nd4jStatus LegacyStatsOp<T>::validateAndExecute(Block<T> &block) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            // we assume that opNuk is either stored in block, or was provided via op constructor
            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            // bias goes as first argument, unlike all other reductions
            bool biasCorrected = false;
            if (block.getIArguments()->size() > 0)
                biasCorrected = block.getIArguments()->at(0) > 0;

            if (block.getIArguments()->size() == 1 || (block.getIArguments()->size() == 2 && block.getIArguments()->at(1) == MAX_INT)) {
                // scalar
                T res = NativeOpExcutioner<T>::execSummaryStatsScalar(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data(),  biasCorrected);
                z->putScalar(0, res);
            } else {
                // dimensions for TAD
                // we should skip first argument here, because it's addressing bias correction
                std::vector<int> dims;
                for (int e = 1; e < block.getIArguments()->size(); e++)
                    dims.emplace_back(block.getIArguments()->at(e));

                if (dims.size() > 1)
                    std::sort(dims.begin(), dims.end());

                REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions requuired for reduction!");

                NativeOpExcutioner<T>::execSummaryStats(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data(), z->getBuffer(), z->getShapeInfo(), dims.data(), (int) dims.size(), biasCorrected);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        template <typename T>
        LegacyStatsOp<T>::LegacyStatsOp() : LegacyOp<T>::LegacyOp(1) {
            //
        }

        template <typename T>
        LegacyStatsOp<T>::LegacyStatsOp(int opNum) : LegacyOp<T>::LegacyOp(1, opNum) {
            //
        }

        /**
        *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
        *   It solely depends on input shape, and requested dimensions
        */
        template <typename T>
        ShapeList *LegacyStatsOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Block<T> &block) {
            auto inShape = inputShape->at(0);

            int *newShape;
            if (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && block.getIArguments()->at(0) == MAX_INT)) {
                // in this case we just return scalar
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);
                newShape[0] = 2;
                newShape[1] = 1;
                newShape[2] = 1;
                newShape[3] = 1;
                newShape[4] = 1;
                newShape[5] = 0;
                newShape[6] = 1;
                newShape[7] = 99;
            } else {
                // in this case we're building proper shape for reduction
                auto array = new NDArray<T>(nullptr, inShape, block.getWorkspace());
                array->triggerAllocationFlag(false, false);

                newShape = array->evalReduceShapeInfo('c', *block.getIArguments());

                delete array;
            }

            return new ShapeList(newShape);
        }

        template class LegacyStatsOp<float>;
        template class LegacyStatsOp<double>;
        template class LegacyStatsOp<float16>;
    }
}