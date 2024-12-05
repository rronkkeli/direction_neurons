#include <zephyr/kernel.h>
#include <math.h>
#include "confusion.h"
#include "adc.h"
#include "central_points.h"
#include "neural_net.h"

const bool USE_NEURAL_NET = false;

/* 
  K-means algorithm should provide 6 center points with
  3 values x,y,z. Let's test measurement system with known
  center points. I.e. x,y,z are supposed to have only values
  1 = down and 2 = up
  
  CP matrix is thus the 6 center points got from K-means algoritm
  teaching process. This should actually come from include file like
  #include "KmeansCenterPoints.h"
  
  And measurements matrix is just fake matrix for testing purpose
  actual measurements are taken from ADC when accelerator is connected.
*/

// How many measurements should be taken when command is given?
const int MEASUREMENT_COUNT = 10000;

/*

*/
int CM[6][6]= {0};
float dense_layer_neurons[1][64];
float output_layer_neurons[1][6];

void printConfusionMatrix(void)
{
	printk("Confusion matrix = \n");
	printk("    cp1    cp2    cp3    cp4    cp5    cp6\n");
	for(int i = 0; i < 6; i++)
	{
		printk("cp%d %d      %d      %d      %d      %d      %d\n", i+1, CM[i][0], CM[i][1], CM[i][2], CM[i][3], CM[i][4], CM[i][5]);
	}
}

void makeHundredFakeClassifications(void)
{
   /*******************************************
   Jos ja toivottavasti kun teet toteutuksen paloissa eli varmistat ensin,
   että etäisyyden laskenta 6 keskipisteeseen toimii ja osaat valita 6 etäisyydestä
   voittajaksi sen lyhyimmän etäisyyden, niin silloin voit käyttää tätä aliohjelmaa
   varmistaaksesi, että etäisuuden laskenta ja luokittelu toimii varmasti tunnetulla
   itse keksimälläsi sensoridatalla ja itse keksimilläsi keskipisteillä.
   *******************************************/
   printk("Make your own implementation for this function if you need this\n");
}

/*
   Takes the intented direction and increments its corresponding
   row in the confusion matrix
*/
void makeOneClassificationAndUpdateConfusionMatrix(int direction)
{
   /**************************************
   Tee toteutus tälle ja voit tietysti muuttaa tämän aliohjelman sellaiseksi,
   että se tekee esim 100 kpl mittauksia tai sitten niin, että tätä funktiota
   kutsutaan 100 kertaa yhden mittauksen ja sen luokittelun tekemiseksi.
   **************************************/
   struct Measurement m;
   int winner;

   for (int i = 0; i < MEASUREMENT_COUNT; i++) {
      m = readADCValue();

      if (USE_NEURAL_NET) {
         printk("C: %d %d %d", m.x, m.y, m.z);
         winner = neuralNetDir(m.x, m.y, m.z);
      } else {
         winner = calculateDistanceToAllCentrePointsAndSelectWinner(m.x, m.y, m.z);
      }

      CM[direction][winner]++;
   }
}

/*
   Calculates the distance to all central points and selects the closest point
   the winner of the point. Takes coordinates of the point as parameters.
*/
int calculateDistanceToAllCentrePointsAndSelectWinner(int x, int y, int z)
{
   int winner = 0;
   float shortest = sqrt(powf(x - cp[0][0], 2.0) + powf(y - cp[0][1], 2.0) + powf(z - cp[0][2], 2.0));

   for (int i = 1; i < 6; i++) {
      float distance = sqrt(powf(x - cp[i][0], 2.0) + powf(y - cp[i][1], 2.0) + powf(z - cp[i][2], 2.0));

      if (distance < shortest) {
         winner = i;
         shortest = distance;
      }
   }

   return winner;
}

void resetConfusionMatrix(void)
{
	for(int i=0;i<6;i++)
	{ 
		for(int j = 0;j<6;j++)
		{
			CM[i][j]=0;
		}
	}
}

int neuralNetDir(int x, int y, int z) {
   // Esikäsittele koordinaatit
   int middle = 0;
   if ((x <= y && x >= z) || (x >= y && x <= z)) {
      middle = x;
   } else if ((y <= x && y >= z) || (y >= x && y <= z)) {
      middle = y;
   }  else if ((z <= x && z >= y) || (z >= x && z <= y)) {
      middle = z;
   }

   x -= middle;
   y -= middle;
   z -= middle;

   resetNeurons();
   // printk("Dense layer first element: %f\n", dense_layer_neurons[0][0]);
   float v[1][3] = {{x, y, z}};
   middlemise(v);
   // neuralNetDense(v);
   // printk("Dense layer first element: %f\n", dense_layer_neurons[0][0]);
   float output[6];
   neuralNetOut(v, output);
   float max = 0;
   int win = 0;

   for (int i = 0; i < 6; i++) {
      if (output_layer_neurons[0][i] > max) {
         win = i;
         max = output_layer_neurons[0][i];
      }
   }

   printk("Winner was %d\n", win);

   return win;
}

void neuralNetOut(float values[1][3], float *output) {
   for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 3; j++) {
         output_layer_neurons[0][i] += (values[0][j] * w_dense[j][i]);
      }

      output_layer_neurons[0][i] += b_dense[i][0];
   }

   softmax(output_layer_neurons[0], 6, output);

   for (int i = 0; i < 6; i++) {
      output_layer_neurons[0][i] = output[i];
   }
}

void resetNeurons() {
   for (int i = 0; i < 6; i++) {
      output_layer_neurons[0][i] = 0.;
   }
}

double sigmoid(double value) {
   float sig = 1.0 / (1.0 + expf(-value));

   printk("Sigmoid result of value %f: %f\n", value, sig);
   return sig;
}

float relu(float x) {
   if (x < 0.0f) {
      return 0.0f;
   } else {
      return x;
   }
}

void middlemise(float x[1][3]) {
   int middle = 2;
   if ((x[0][0] <= x[0][1] && x[0][0] >= x[0][2]) || (x[0][0] >= x[0][1] && x[0][0] <= x[0][2])) {
      middle = 0;
   } else if ((x[0][1] <= x[0][0] && x[0][1] >= x[0][2]) || (x[0][1] >= x[0][0] && x[0][1] <= x[0][2])) {
      middle = 1;
   }  else if ((x[0][2] <= x[0][0] && x[0][2] >= x[0][1]) || (x[0][2] >= x[0][0] && x[0][2] <= x[0][1])) {
      middle = 2;
   }

   x[0][0] -= x[0][middle];
   x[0][1] -= x[0][middle];
   x[0][2] -= x[0][middle];
}

void softmax(float *input, int input_len, float *output) {
  float max_val = input[0];
  for (int i = 1; i < input_len; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  float sum_exp = 0.0f;
  for (int i = 0; i < input_len; i++) {
    output[i] = expf(input[i] - max_val); // Subtract max for numerical stability
    sum_exp += output[i];
  }

  for (int i = 0; i < input_len; i++) {
    output[i] /= sum_exp;
  }
}