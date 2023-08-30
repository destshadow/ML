#define NN_IMPLEMENTATION
#include "nn.h"


int main(){
    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);
    NN_PRINT(nn);


    return 0;
}

//EP 5 12.30