//
//  matrixfunc.h
//  net
//
//  Created by Filippo Valle on 16/03/2018.
//

#ifndef matrixfunc_h
#define matrixfunc_h

#include <math.h>
#include <iostream>

using std::vector;

namespace functions{
    double step(double x){
        return (x > 0?1:-1);
    }

    double sigmoid (double x) {

        /*  Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
         Input: m1, a vector.
         Output: 1/(1 + e^-x) for every element of the input matrix m1.
         */

        double output = 1 / (1 + exp(-x));

        return output;
    }

    double sigmoid_d(double x){
        return  x * (1 - x);
    }
}

#endif /* matrixfunc_h */
