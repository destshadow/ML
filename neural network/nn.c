#include <time.h>
#define NN_IMPLEMENTATION  //da accesso alla parte di implementazione nel file h
#include "nn.h"

/*
float td_sum[] = {
    0, 0,   0, 0,   0, 0,
    0, 0,   0, 1,   0, 1,
    0, 1,   0, 1,   1, 0,
    0, 1,   1, 0,   1, 1,
};*/

float td_xor[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

float td_or[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

int main(){
    srand(time(0));


/*  float id_data[4] = {1, 0, 0, 1};
    Mat b = {.rows = 2, .cols = 2, .es = id_data};
*/

    float *td = td_or;

    size_t stride = 3;
    size_t n = 4;//sizeof(td) / sizeof(td[0]) / stride;

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td,
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    };

    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);


    float rate = 1;


    printf("cost of nn: %f\n", nn_cost(nn, ti, to));
    for(size_t i = 0; i < 5000; ++i){

        nn_backprop(nn, g, ti, to);

        //NN_PRINT(g);
        nn_learn(nn, g, rate);
        printf("%zu: cost of nn: %f\n", i, nn_cost(nn, ti, to));
    }

    /*Mat row = mat_row(ti, 1);
    MAT_PRINT(row);
    mat_copy(NN_INPUT(nn) , row);
    nn_forwawrd(nn);
    MAT_PRINT( NN_OUTPUT(nn));*/

    NN_PRINT(nn);

    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forwawrd(nn);
            printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }

    }

    return 0;
}

//time stamp ep 3 3.20.34