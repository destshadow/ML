#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


typedef struct{
    float or_w1;
    float or_w2;
    float or_b;

    float and_w1;
    float and_w2;
    float and_b;

    float nand_w1;
    float nand_w2;
    float nand_b;
}Xor;

//limita il valore tra 0 e 1 (approccia 0 e 1 all'infinito) (tipo scalino ma smooth)
float sigmoidf(float x){ 
    return 1.f / (1.f + expf(-x));
}


float forward(Xor m, float x1, float x2){
    float a = sigmoidf(m.or_w1 * x1 +  m.or_w2 * x2 + m.or_b);
    float b = sigmoidf(m.nand_w1 * x1 +  m.nand_w2 * x2 + m.nand_b);
    return sigmoidf(a * m.and_w1 + b * m.and_w2 + m.and_b);
}

typedef float sample[3];

sample xor_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

//posso usare anche i modelli precedenti in questa iterazione di AI
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

sample nor_train[] = {
    {0, 0, 1},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 0},
};

sample *train = xor_train;
size_t train_count = 4;

float Cost_function(Xor m){
    float result = 0.0f;

    //Funzione di costo o loss più vicina è a 0 meglio è
    for(size_t i = 0; i < train_count; i++){
        //inputs
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2);        //neurone o perceptron
        float d = y - train[i][2];
        result += d * d; //distanza amplificata per il quadrato e se è negativa non importa
        //printf("actual: %f, expected: %f\n", y, train[i][1]);
    }

    return result /= train_count;
}

float rand_float(){
    return (float) rand() / (float) RAND_MAX;
}

Xor rand_xor(){
    Xor m;
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.or_b = rand_float() ;

    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.and_b = rand_float();

    m.nand_w1 = rand_float();
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();

    return m;
}

void print_xor(Xor m){
    printf("or_w1: %f\n", m.or_w1) ;
    printf("or_w2: %f\n", m.or_w2) ;
    printf("or_b: %f\n", m.or_b);

    printf("and_w1: %f\n", m.and_w1);
    printf("and_w2: %f\n", m.and_w2);
    printf("and_b: %f\n", m.and_b);

    printf("nand_w1: %f\n", m.nand_w1);
    printf("nand_w2: %f\n", m.nand_w2) ;
    printf("nand_b: %f\n", m.nand_b);
}

//ritorna il gradiente
Xor finite_diff(Xor m, float eps){

    Xor g;
    float cost = Cost_function(m);
    float saved;

    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (Cost_function(m) - cost) / eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (Cost_function(m) - cost) / eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (Cost_function(m) - cost) / eps;
    m.or_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (Cost_function(m) - cost) / eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (Cost_function(m) - cost) / eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (Cost_function(m) - cost) / eps;
    m.and_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (Cost_function(m) - cost) / eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (Cost_function(m) - cost) / eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (Cost_function(m) - cost) / eps;
    m.nand_b = saved;

    return g;
}

Xor apply_diff(Xor m, Xor g, float rate){
    m.or_w1 -= rate * g.or_w1;
    m.or_w2 -= rate * g.or_w2;
    m.or_b -= rate * g.or_b;

    m.and_w1 -= rate * g.and_w1;
    m.and_w2 -= rate * g.and_w2;
    m.and_b -= rate * g.and_b;

    m.nand_w1 -= rate * g.nand_w1;
    m.nand_w2 -= rate * g.nand_w2;
    m.nand_b -= rate * g.nand_b;

    return m;
}


int main(){
    srand(time(0));
    Xor m = rand_xor();

    float eps = 1e-1;
    float rate = 1e-1;

    for(size_t i = 0; i < 100*1000; i++){
        Xor g = finite_diff(m, eps);
        m = apply_diff(m,g,rate);
        //printf("cost = %f\n", Cost_function(m));
    }
    printf("cost = %f\n", Cost_function(m));

    printf("---------------------------------------------------\n");

    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            printf("%zu XOR %zu = %f\n", i, j, forward(m, i, j));
        }
    }

    printf("---------------------------------------------------\n");

    printf("\"OR\" neuron:\n");
    //ora controlliamo il pezzo in cui proviamo a fare l'OR secondo il nostro schema e abbiamo trovato che l'AI al posto dell'or ci esegue un and
    //ha trovato una formula diversa per creare uno xor (senza generare numeri random)
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            printf("%zu | %zu = %f\n", i, j, sigmoidf(m.or_w1 * i +  m.or_w2 * j + m.or_b));
        }
    }

    printf("---------------------------------------------------\n");
    printf("\"NAND\" neuron:\n");
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            printf("~(%zu & %zu) = %f\n", i, j, sigmoidf(m.nand_w1 * i +  m.nand_w2 * j + m.nand_b));
        }
    }

    printf("---------------------------------------------------\n");
    printf("\"AND\" neuron:\n");
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            printf("%zu & %zu = %f\n", i, j, sigmoidf(m.and_w1 * i +  m.and_w2 * j + m.nand_b));
        }
    }

    /*Lasciamo che l'AI faccia quello che voglia per avere comunque il risultato corretto alla fine
    x   -> OR
                } -> AND -> XOR
    y   -> NAND

    */

    return 0;
}

//time stamp 2.05.27
//https://www.youtube.com/watch?v=PGSba51aRYU