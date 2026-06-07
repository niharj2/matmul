#include <vector>

std::vector<std::vector<int>> matrix_multplication(std::vector<std::vector<int>> M, std::vector<std::vector<int>> N) {
    std::vector<std::vector<int>> product(M.size(), std::vector<int>(M.size())); 

    for (int i = 0; i < M.size(); ++i) {
        for (int j = 0; j < N.size(); ++j) {
            int sum = 0; 
            for (int k = 0; k < N.size(); ++k) {
                int a = M[i][k]; 
                int b = N[k][j]; 
                sum += a * b;
            }
            product[i][j] = sum;
        }
    }
    return product;
}


