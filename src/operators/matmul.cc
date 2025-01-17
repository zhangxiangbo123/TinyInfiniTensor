#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A_dim = inputs[0]->getDims();
        auto A_rank = inputs[0]->getRank();
        auto B_dim = inputs[1]->getDims();
        auto B_rank = inputs[1]->getRank();       
        
        auto res_rank = std::max(A_rank, B_rank);
        Shape res(res_rank);
        
        // 应至少为二位矩阵
        if (A_rank <2 || B_rank<2){
            return std::nullopt; 
        }

        // 双向广播
        int i = int(A_rank) -3;
        int j = int(B_rank) -3;

        for(; i >=0 && j>=0; i--, j--){
            if (A_dim[i] == B_dim[j]){
                res[std::max(i,j)] = A_dim[i];
            } else if (A_dim[i] == 1){
                res[std::max(i,j)] = B_dim[j];
            } else if (B_dim[j] == 1){
                res[std::max(i,j)] = A_dim[i];
            } else {
                return std::nullopt;
            }
        }

        for (; i >=0; i--){
            res[i] = A_dim[i];
        }

        for (; j >=0; j--){
            res[j] = B_dim[j];
        }

        // 检查是否满足矩阵乘法
        auto A_r = A_dim[A_rank-1];
        auto A_c = A_dim[A_rank-2];
        auto B_r = B_dim[A_rank-1];
        auto B_c = B_dim[A_rank-2];

        if (this->transA){
            std::swap(A_r, A_c);
        }

        if (this->transB){
            std::swap(B_r, B_c);
        }

        if (A_r != B_c){
            return std::nullopt;
        }

        res[res_rank - 2] = A_c;
        res[res_rank - 1] = B_r;

        // 检查是否有偏置矩阵C
        if (inputs.size() < 3){
            return {{res}};
        }

        auto C_dim = inputs[2]->getDims();
        auto C_rank = inputs[2]->getRank();  

        if (C_rank < 2){
            return std::nullopt;
        }

        // 对结果res和C进行广播
        int m = int(res_rank) - 1;
        int n = int(C_rank) - 1;

        for(; n>=0; m--, n--){
            if (res[m] != C_dim[n] && C_dim[n]!=1){
                return std::nullopt;
            }
        }
       
        return {{res}};
    }

} // namespace infini