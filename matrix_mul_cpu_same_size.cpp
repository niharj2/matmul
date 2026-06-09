#include <vector>
// The input matrices are 2D 
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

// Assums all matrices are 1D. This is actually the preferred way and is faster
std::vector<int> matrix_multplication_1D(std::vector<int> M, std::vector<int>N, int Width) {
    std::vector<int>Product(Width * Width); 

    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            int sum = 0;
            for (int k = 0; k < Width; ++k) {
                int a = M[i * Width + k]; 
                int b = N[k * Width + j];
                sum += a * b;
            }
            Product[i * M.size() + j] = sum;
        }
    }
    return Product; 
}


