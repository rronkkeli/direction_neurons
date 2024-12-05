#ifndef CONFUSION_MATRIX_H
#define CONFUSION_MATRIX_H


void printConfusionMatrix(void);
void makeHundredFakeClassifications(void);
void makeOneClassificationAndUpdateConfusionMatrix(int);
int calculateDistanceToAllCentrePointsAndSelectWinner(int,int,int);
void resetConfusionMatrix(void);
int neuralNetDir(int, int, int);
void neuralNetOut(float[1][3], float*);
double sigmoid(double);
float relu(float);
void resetNeurons(void);
void middlemise(float[1][3]);
void softmax(float*, int, float*);

#endif
