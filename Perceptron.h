//
// Created by Filippo Valle on 17/03/2018.
//

#ifndef NET_NEURON_H
#define NET_NEURON_H

#include <iostream>
#include <utility>
#include <vector>
#include <numeric>
#include <cmath>

#include "matrixfunc.h"

#include "pcg_random.hpp"

using std::vector;

template <class datatype>
class Perceptron {
public:
    Perceptron(const vector<datatype> &fX, vector<datatype> fy, unsigned int numOfData, uint32_t seed=42);


    void fit();
    int infere(vector<datatype> X);

private:
    bool fTrained;

    vector<datatype> fW_delta;
    vector<datatype> fW;
    vector<datatype> fX;
    vector<datatype> fy;
    unsigned long fNumOfData;
    unsigned long fNumOfFeatures;

    const double fSigma = 0.01;

    double predict(vector<datatype> X);
    bool isInputSizeOk(vector<datatype> X);
};


template<class datatype>
Perceptron<datatype>::Perceptron(const vector<datatype> &X, vector<datatype> y, unsigned int numOfData, uint32_t seed):fX(X), fy(std::move(y)), fNumOfData(numOfData) {
    pcg32_fast myRng(seed);
    std::uniform_int_distribution<datatype> distribution(0.,1.);


    fNumOfFeatures=fX.size()/fNumOfData;

    //add vapnick dimension
    for(int i=0;i<fNumOfFeatures+1;i++){
        fW.push_back(distribution(myRng));
    }

    fW_delta.reserve(fNumOfFeatures+1);

    fTrained = false;
}

template<class datatype>
void Perceptron<datatype>::fit() {
    bool learnt=false;

    for (unsigned step = 0; step < 5000; ++step) {
        if(learnt) {
            //std::cout<<"step:"<<step<<std::endl;
            break;
        }

        int iInput=0;
        for(auto iterInput=fX.begin(); iterInput < fX.end()-fNumOfFeatures+1;iterInput+=fNumOfFeatures) {
            learnt= true;
            auto input=vector<datatype>(iterInput,iterInput+fNumOfData);
            input.insert(input.begin(),1);

            double pred=functions::step(std::inner_product(input.begin(),input.end(),fW.begin(),0.));

            datatype y = fy[iInput];
            //algo works for y=+-1, otherwise does not converge
            if(y==0) y=-1;

            double pred_error = fabs(y - pred);
            //double pred_delta = pred_error * functions::sigmoid_d(pred);
           //printf("\nI:%d pred: %f, y:%f, delta:%f\n",iInput, pred,y,pred_error);

            if (pred_error > fSigma) {
                learnt = false;
                std::copy(input.begin(),input.end(),fW_delta.begin());
                //std::for_each(fW_delta.begin(),fW_delta.end(),[y](datatype &x){x*=y;});
                for (int i = 0; i < fNumOfFeatures + 1; i++) {
                    fW_delta[i] *= y;
                }

                for (int i = 0; i < fNumOfFeatures + 1; i++) {
                    fW[i] += fW_delta[i];
                }

                break;
            }

            iInput++;
        }
    };

    fTrained = true;
}


template<class datatype>
double Perceptron<datatype>::predict(vector<datatype> X) {
    if (!fTrained) fit();
    //-1 is due to vapnik dimension
    if ((X.size() - 1) != fNumOfFeatures) {
        printf("\n*****\nwrong data size:\n\tgiven:%lu\n\texpected:%lu", X.size()-1, fNumOfFeatures);
        return -1;
    } else {
        return functions::step(std::inner_product(X.begin(), X.end(), fW.begin(), 0.0));
    }
}

template<class datatype>
int Perceptron<datatype>::infere(vector<datatype> X) {
    X.insert(X.begin(), 1);
    //printf("%f ",predict(X));
    if (!isInputSizeOk(X)) {
        return -1;
    } else {
        return (predict(X) == 1 ? 1 : 0);
    }
}

template<class datatype>
bool Perceptron<datatype>::isInputSizeOk(vector<datatype> X) {
    return (X.size()-1)==fNumOfFeatures;
}

#endif //NET_NEURON_H
