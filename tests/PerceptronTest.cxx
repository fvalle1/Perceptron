#include "gtest/gtest.h"
#include "Perceptron.h"

using std::vector;

vector<float> X {5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 6.2, 3.4, 5.4, 2.3, 5.9, 3.0, 5.1, 1.8};

vector<float> y { 0,0,1,1 };

TEST(perc__Test, perc__Test_simple_predict_Test_zero) {
    Perceptron<float> p(X, y, 4);
    p.fit();

    vector<float> testInput{5.2, 3.4, 1.5, 0.3};
    EXPECT_EQ(p.infere(testInput), y[0]);
}

TEST(perc__Test,perc__Test_simple_predict_Test_one) {
    Perceptron<float> p(X, y, 4);
    p.fit();

    vector<float> testInput{5.9, 3.0, 5, 1.7};
    EXPECT_EQ(p.infere(testInput), y[3]);
}

TEST(perc_input__Test, forget_training_Test){
    Perceptron<float> p(X,y,4);

    vector<float> testInput {5.2,3.4,1.5,0.3};
    EXPECT_EQ(p.infere(testInput),y[0]);
}

TEST(perc_input__Test, wrong_data_size_Test){
    Perceptron<float> p(X,y,4);
    p.fit();

    vector<float> testInput {5.2,3.4,1.5,0.3,4.6};
    EXPECT_EQ(p.infere(testInput),-1);
}