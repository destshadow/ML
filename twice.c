#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//il primo è l'input l'altro è quello che si aspetta
float train[][2] ={
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

//ritorna il numero di elementi
#define train_count (sizeof(train) / sizeof(train[0]))

//random da 0 a 1
float rand_float(){
    return (float) rand() / (float) RAND_MAX;
}

//quanto performa il nostro modello
float Cost_function(float w, float bias){
    float result = 0.0f;

    //Funzione di costo o loss più vicina è a 0 meglio è
    for(size_t i = 0; i < train_count; i++){
        //input
        float x = train[i][0];
        float y = x * w + bias;        //neurone o perceptron
        float d = y - train[i][1];
        result += d * d; //distanza amplificata per il quadrato e se è negativa non importa
        //printf("actual: %f, expected: %f\n", y, train[i][1]);
    }

    return result /= train_count;
}


int main(){
    //srand(time(0));
    srand(69);

    float esp = 1e-3; //valore molto piccolo
    float w = rand_float() * 10.0f; // valori da 0 a 10        nel nostro caso w deve essere uguale a 2 cioè moltiplica per 2 i valori
    float bias = rand_float() * 5.0f;

    float rate = 1e-3;

    for(int i = 0; i < 500; ++i){
        //finite difference non è una derivata normale è giusto per testing
        float c = Cost_function(w, bias);
        float distance_cost = (Cost_function(w + esp, bias) - c) / esp;
        float distance_bias = (Cost_function(w, bias + esp) - c) / esp;

        w -= rate * distance_cost;
        bias -= rate * distance_bias;

        printf("costo: %f, w: %f, bias: %f \n", Cost_function(w, bias), w, bias);
    }

    printf("%s", "---------------------------\n");
    printf("w: %f, bias: %f \n", w, bias);

    /*printf("quanto performa il nostro modello: %f\n", Cost_function(w));
    printf("quanto performa il nostro modello: %f\n", Cost_function(w - esp));
    printf("quanto performa il nostro modello: %f\n", Cost_function(w - esp * 2));*/

    return 0;
}

//time stamp: 