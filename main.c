#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <cblas.h>

#define k 2
#define n 64
#define N 12000

float simulation_weights[k][n] = {0};
float model_weights[k][n] = {0};

float challenges[N][n] = {0};
float responses[N] = {0};

float total_model_val[N] = {0};
float model_vals[N][k] = {0};

float eta_minus = .5;
float eta_plus = 1.2;

float delta_max = 10e1;
float delta_min = 10e-4;

float grad[k][n] = {0};
float last_grad[k][n] = {0};

float delta[k][n] = {0};
float last_delta[k][n] = {0};

float delta_weight[k][n] = {0};
float last_delta_weight[k][n] = {0};

// ADAM optimization
float m[k][n] = {0},
        u[k][n] = {0};


float gaussian(void)
{
    float x = random() / (float)RAND_MAX,
          y = random() / (float)RAND_MAX;
    return sqrtf(-2 * logf(x)) * cosf(2 * (float)M_PI * y);
}

void init_weights(float weights[k][n]) {
    for (int j=0;j<k;j++)
        for (int i=0;i<n;i++)
            weights[j][i] = gaussian();
}

void init_challenges(float* challenges, int length) {
    float *target = challenges;
    int total_size = length * n;
    for (int m = 0; m <total_size; m++) {
        if (random() % 256 >= 128)
            *target++ = -1;
        else
            *target++ = 1;
    }
}

static float val(float challenge[n], float weights[k][n], int skip) {
    float running_product = 1;
    for (int j=0; j < k; j++) {
        if (j == skip) continue;
        float running_sum = 0;
        for (int i=0; i < n; i++) {
            if (challenge[i] < 0)
                running_sum += -weights[j][i];
            else
                running_sum += weights[j][i];
        }
        running_product *= running_sum;
    }
    return running_product;
}

static float accuracy() {
    float challenges[1000][n] = {0};
    init_challenges((float*)challenges, 1000);

    int correct = 0;
    for (int m = 0; m < 1000; m++)
        if (val(challenges[m], model_weights, -1) * val(challenges[m], simulation_weights, -1) > 0)
            correct++;

    return (float)correct / 1000;
}

float norm(float grad[k][n]) {
    float sum_of_squares = 0;
    for (int j = 0; j < k; j++)
        for (int i = 0; i < n; i++)
            sum_of_squares += grad[j][i] * grad[j][i];
    return sqrtf(sum_of_squares);
}

static inline float sgn(float x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

static inline float min(float a, float b) {
    if (a < b) return a;
    return b;
}

static inline float max(float a, float b) {
    if (a < b) return b;
    return a;
}

static inline void eval_model() {
    for (int m = 0; m < N; m++) {
        for (int j=0; j < k; j++) {
            model_vals[m][j] = cblas_sdot(n, challenges[m], 1, model_weights[j], 1);
        }
        total_model_val[m] = 1;
        for (int j = 0; j < k; j++)
            total_model_val[m] *= model_vals[m][j];
    }
}

static inline void gradient() {
    float two_by_one_plus_expf_minus_total_model_val[N];
    for (int m = 0; m < N; m++)
        two_by_one_plus_expf_minus_total_model_val[m] = 2 / (1 + expf(-total_model_val[m])) - 1 - responses[m];

    for (int j = 0; j < k; j++) {
        float model_val_without_j[N];
        float gradient_magnitude[N];
        for (int m = 0; m < N; m++) {
            model_val_without_j[m] = total_model_val[m] / model_vals[m][j];
            gradient_magnitude[m] = two_by_one_plus_expf_minus_total_model_val[m] *
                                    model_val_without_j[m];
        }

        for (int i = 0; i < n; i++) {
            grad[j][i] = 0;
            for (int m = 0; m < N; m++) {
                if (challenges[m][i] < 0)
                    grad[j][i] += gradient_magnitude[m];
                else
                    grad[j][i] -= gradient_magnitude[m];
            }
        }
    }
}

void optimize_adam(int round) {
    float alpha = 30, beta1 = 0.9, beta2 = 0.999;
    float grad_norm = norm(grad);

    // ADAM optimization
    for (int j = 0; j < k; j++)
        for (int i = 0; i < n; i++) {
            m[j][i] = beta1 * m[j][i] + (1 - beta1) * grad[j][i];
            u[j][i] = max(beta2 * u[j][i], grad_norm);
            model_weights[j][i] += alpha / (1 - powf(beta1, round + 1)) * m[j][i] / u[j][i];
        }

    // debug output begin
//    int jj = 0, ii = 12;
//    printf("%f\t%f\t%f\t%f\n",
//           m[jj][ii]/u[jj][ii],
//           alpha / (1 - powf(beta1, round + 1)),
//           m[jj][ii]/u[jj][ii] * alpha / (1 - powf(beta1, round + 1)),
//           model_weights[jj][ii]
//    );
    // debug output end
}

void optimize_rprop() {
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < n; i++) {
            float indicator = last_grad[j][i] * grad[j][i];
            if (indicator > 0) {
                delta[j][i] = min(last_delta[j][i] * eta_plus, delta_max);
                delta_weight[j][i] = -sgn(grad[j][i]) * delta[j][i];
                model_weights[j][i] -= delta_weight[j][i];
            } else if (indicator < 0) {
                delta[j][i] = max(last_delta[j][i] * eta_minus, delta_min);
                model_weights[j][i] += last_delta_weight[j][i];
                grad[j][i] = 0;
            } else {
                delta_weight[j][i] = -sgn(grad[j][i]) * delta[j][i];
                model_weights[j][i] -= delta_weight[j][i];
            }
        }
    }

    memcpy(last_grad, grad, sizeof(last_grad)); // TODO just switch pointers?
    memcpy(last_delta, delta, sizeof(last_delta));
    memcpy(last_delta_weight, delta_weight, sizeof(last_delta_weight));
}

void learn(int optimizer) {
    for (int j = 0; j < k; j++)
        for (int i = 0; i < n; i++) {
            last_grad[j][i] = 1;
            last_delta[j][i] = .1;
            delta[j][i] = .1;
        }

    if (optimizer == 0) printf("rel.\tscale\tstep\ttotal\n");
    for (int round = 0; round < 794; round++) {
        eval_model();
        gradient();
        if (optimizer == 0)
            optimize_adam(round);
        else
            optimize_rprop();


//        float acc = accuracy(simulation_weights, model_weights);
//        printf("%i, %f, %f, %f\n", rounds, acc, norm(grad), norm(delta_weight));
//        if (acc >= .99) return;
    }
}


int main() {
    init_weights(simulation_weights);
    init_weights(model_weights);

    init_challenges(&challenges[0][0], N);
    for (int m = 0; m < N; m++)
        responses[m] = sgn(val(challenges[m], simulation_weights, -1));

    learn(0);
    printf("ADAM accuracy: %f\n", accuracy());
    init_weights(model_weights);
    learn(1);
    printf("RPROP accuracy: %f\n", accuracy());

    return 0;
}
