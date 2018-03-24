#include <iostream>
#include <vector>
#include "Perceptron.h"

#include "Perceptron.h"

using std::vector;

vector<float> X {5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 6.2, 3.4, 5.4, 2.3, 5.9, 3.0, 5.1, 1.8};

vector<float> y { 0,0,1,1 };

int main(){
    Perceptron<float> p(X, y, 4);
    p.fit();

    vector<float> testInput{5.2, 3.4, 1.5, 0.3};
    printf("predict: %d",p.infere(testInput));


}