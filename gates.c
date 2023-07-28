#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/*and or nand porte logiche create tutte con un neurone
https://www.youtube.com/watch?v=PGSba51aRYU

*/

//limita il valore tra 0 e 1 (approccia 0 e 1 all'infinito) (tipo scalino ma smooth)
float sigmoidf(float x){ 
    return 1.f / (1.f + expf(-x));
}
/*
for testing (in main)
    for(float x = -10.f; x <= 10.f; x += 1.f){
        printf("%f => %f\n", x, sigmoidf(x));
    }
*/

typedef float sample[3];

//OR gate
sample or_train[] ={
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

sample and_train[] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1},
};

sample nand_train[] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

sample *train = nand_train;
size_t train_count = 4;

//ritorna il numero di elementi in una matrice
//#define train_count (sizeof(train) / sizeof(train[0]))

//quanto performa il nostro modello
float Cost_function(float w1, float w2, float bias){
    float result = 0.0f;

    //Funzione di costo o loss più vicina è a 0 meglio è
    for(size_t i = 0; i < train_count; i++){
        //inputs
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1 * w1 + x2 * w2 + bias);        //neurone o perceptron
        float d = y - train[i][2];
        result += d * d; //distanza amplificata per il quadrato e se è negativa non importa
        //printf("actual: %f, expected: %f\n", y, train[i][1]);
    }

    return result /= train_count;
}

//random da 0 a 1
float rand_float(){
    return (float) rand() / (float) RAND_MAX;
}

int main(){
    srand(time(0));

    //pesi iniziali
    float w1 = rand_float();
    float w2 = rand_float();
    float eps = 1e-1;
    float rate = 1e-1;
    float bias = rand_float();

    //training
    for(size_t i = 0; i < 2000*1000; ++i){

        float c = Cost_function(w1, w2, bias);

        //printf("w1: %f, w2: %f, bias: %f, c: %f\n", w1, w2, bias, c);
        
        float dw1 = (Cost_function(w1 + eps, w2, bias) - c) / eps;
        float dw2 = (Cost_function(w1, w2 + eps, bias) - c) / eps;
        float db = (Cost_function(w1, w2, bias + eps) - c) / eps;
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        bias -= rate * db;
    }

    //printf("w1: %f, w2: %f, bias: %f, c: %f\n", w1, w2, bias, Cost_function(w1, w2, bias));

    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            printf("%zu OR %zu = %f\n", i, j, sigmoidf(i * w1 + j * w2 + bias) );
        }
    }

    return 0;
}

//time stamp: 1.28.12