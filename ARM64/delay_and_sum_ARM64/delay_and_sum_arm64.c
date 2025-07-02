#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>

#define NUM_SENSORES 12
#define NUM_PARES 132
#define NUM_MUESTRAS 4096
#define VELOCIDAD_ONDA 5000
#define FRECUENCIA_MUESTREO 12.5E6
#define ANCHO 0.5
#define ALTO 0.70

// Variables globales para GRID (configurables por línea de comandos)
int GRID_X;
int GRID_Y;

typedef struct {
    double x_grid[300][300];
    double y_grid[300][300];
    double distancias[300][300][NUM_SENSORES];
    double indices_dano[300][300];
} Cuadricula;

double sensores_x[NUM_SENSORES] = {0, 0.009, 0.018, 0.027, 0.036, 0.045, 0.054, 0.063, 0.072, 0.081, 0.090, 0.099};
double sensores_y[NUM_SENSORES] = {0};

void cargarDatos(const char* filename, double (*data)[NUM_MUESTRAS]);
void hilbert_transform(double* signal, fftw_complex* hilbert_result, int num_samples);
void calcularIndiceDeDano(Cuadricula* cuadriculaCon, double (*dataConDano)[NUM_MUESTRAS], fftw_complex (*hilbertConDano)[NUM_MUESTRAS]);
void inicializarCuadricula(Cuadricula* cuadricula);
void calcularDistancias(Cuadricula* cuadricula);
void guardarIndiceDeDanoEnCSV(Cuadricula* cuadricula, const char* filename);
double calcularMedia(double* tiempos, int n);
double calcularDesviacion(double* tiempos, int n, double media);

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Uso: %s <num_iter> <GRID_X> <GRID_Y>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int num_iter = atoi(argv[1]);
    GRID_X = atoi(argv[2]);
    GRID_Y = atoi(argv[3]);

    if (num_iter <= 0 || GRID_X <= 0 || GRID_Y <= 0 || GRID_X > 300 || GRID_Y > 300) {
        fprintf(stderr, "Error: Argumentos no válidos.\n");
        return EXIT_FAILURE;
    }

    printf("Ejecutando %d iteraciones con GRID_X=%d y GRID_Y=%d\n", num_iter, GRID_X, GRID_Y);

    double (*dataConDano)[NUM_MUESTRAS] = malloc(sizeof(double[NUM_MUESTRAS]) * NUM_PARES);
    fftw_complex (*hilbertConDano)[NUM_MUESTRAS] = fftw_malloc(sizeof(fftw_complex[NUM_MUESTRAS]) * NUM_PARES);
    Cuadricula* cuadriculaCon = malloc(sizeof(Cuadricula));

    if (!dataConDano || !hilbertConDano || !cuadriculaCon) {
        fprintf(stderr, "Error al reservar memoria.\n");
        return EXIT_FAILURE;
    }

    inicializarCuadricula(cuadriculaCon);
    calcularDistancias(cuadriculaCon);

    double* tiempos = malloc(sizeof(double) * num_iter);

    for (int iter = 0; iter < num_iter; iter++) {
        clock_t start_time = clock();
        cargarDatos("data_damage_scaled_processed.csv", dataConDano);
        for (int i = 0; i < NUM_PARES; i++) {
            hilbert_transform(dataConDano[i], hilbertConDano[i], NUM_MUESTRAS);
        }
        calcularIndiceDeDano(cuadriculaCon, dataConDano, hilbertConDano);
        clock_t end_time = clock();
        tiempos[iter] = ((double)(end_time - start_time) / CLOCKS_PER_SEC)*1000;
        //printf("Iteración %d: %.8f segundos\n", iter+1, tiempos[iter]);
    }

    guardarIndiceDeDanoEnCSV(cuadriculaCon, "70arm64.csv");

    double media = calcularMedia(tiempos, num_iter);
    double desviacion = calcularDesviacion(tiempos, num_iter, media);
    printf("\nMedia de tiempos: %.8f ms\nDesviación típica: %.8f ms\n", media, desviacion);

    free(dataConDano);
    fftw_free(hilbertConDano);
    free(cuadriculaCon);
    free(tiempos);

    return 0;
}

double calcularMedia(double* tiempos, int n) {
    double suma = 0.0;
    for (int i = 0; i < n; i++) suma += tiempos[i];
    return suma / n;
}

double calcularDesviacion(double* tiempos, int n, double media) {
    double suma = 0.0;
    for (int i = 0; i < n; i++) suma += (tiempos[i] - media) * (tiempos[i] - media);
    return sqrt(suma / n);
}

void cargarDatos(const char* filename, double (*data)[NUM_MUESTRAS]) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error al abrir %s\n", filename);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < NUM_PARES; i++) {
        for (int j = 0; j < NUM_MUESTRAS; j++) {
            if (fscanf(file, "%lf,", &data[i][j]) != 1) {
                fprintf(stderr, "Error leyendo dato %d,%d\n", i, j);
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}

void hilbert_transform(double* signal, fftw_complex* hilbert_result, int num_samples) {
    fftw_complex* freq_domain = fftw_malloc(sizeof(fftw_complex) * num_samples);
    fftw_complex* time_domain = fftw_malloc(sizeof(fftw_complex) * num_samples);
    fftw_plan forward = fftw_plan_dft_1d(num_samples, time_domain, freq_domain, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan backward = fftw_plan_dft_1d(num_samples, freq_domain, hilbert_result, FFTW_BACKWARD, FFTW_ESTIMATE);

    for (int i = 0; i < num_samples; i++) {
        time_domain[i][0] = signal[i];
        time_domain[i][1] = 0.0;
    }

    fftw_execute(forward);

    for (int k = 0; k < num_samples; k++) {
        if (k == 0 || k == num_samples / 2) {
            freq_domain[k][0] = freq_domain[k][1] = 0.0;
        } else if (k < num_samples / 2) {
            freq_domain[k][0] *= -1.0;
            freq_domain[k][1] *= -1.0;
        }
    }

    fftw_execute(backward);

    for (int i = 0; i < num_samples; i++) {
        hilbert_result[i][0] /= num_samples;
        hilbert_result[i][1] /= num_samples;
    }

    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(freq_domain);
    fftw_free(time_domain);
}

void inicializarCuadricula(Cuadricula* cuadricula) {
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            cuadricula->x_grid[i][j] = -0.3 + j * ((ANCHO + 0.3) / (GRID_X - 1));
            cuadricula->y_grid[i][j] = -0.05 + i * ((ALTO + 0.05) / (GRID_Y - 1));
            for (int k = 0; k < NUM_SENSORES; k++) {
                double dx = cuadricula->x_grid[i][j] - sensores_x[k];
                double dy = cuadricula->y_grid[i][j];
            }
            cuadricula->indices_dano[i][j] = 0.0;
        }
    }
}

void calcularDistancias(Cuadricula* cuadricula) {
    for (int q = 0; q < GRID_Y; q++) {
        for (int w = 0; w < GRID_X; w++) {
            for (int s = 0; s < NUM_SENSORES; s++) {
                double x = cuadricula->x_grid[q][w];
                double y = cuadricula->y_grid[q][w];
                double sx = sensores_x[s];
                cuadricula->distancias[q][w][s] = sqrt(pow(x - sx, 2) + pow(y, 2));
            }
        }
    }
}

void calcularIndiceDeDano(Cuadricula* cuadriculaCon, double (*dataConDano)[NUM_MUESTRAS], fftw_complex (*hilbertConDano)[NUM_MUESTRAS]) {
    double Ts = 1.0 / FRECUENCIA_MUESTREO;
    for (int q = 0; q < GRID_Y; q++) {
        for (int w = 0; w < GRID_X; w++) {
            fftw_complex D_O_con = {0.0, 0.0};
            for (int E = 1; E <= NUM_SENSORES; E++) {
                for (int R = 1; R <= NUM_SENSORES; R++) {
                    if (E != R) {
                        int index = (E - 1) * (NUM_SENSORES - 1) + (R - 1) - (R > E);
                        double de = cuadriculaCon->distancias[q][w][E - 1];
                        double dr = cuadriculaCon->distancias[q][w][R - 1];
                        double t0 = ((de + dr) / VELOCIDAD_ONDA) / Ts;
                        int n0 = (int)ceil(t0);
                        if (n0 < NUM_MUESTRAS && n0 >= 0) {
                            double realCon = dataConDano[index][n0];
                            double imagCon = hilbertConDano[index][n0][1];
                            fftw_complex H = {realCon, imagCon};
                            double factor = sqrt(de * dr);
                            D_O_con[0] += factor * H[0];
                            D_O_con[1] += factor * H[1];
                        }
                    }
                }
            }
            cuadriculaCon->indices_dano[q][w] = sqrt(D_O_con[0]*D_O_con[0] + D_O_con[1]*D_O_con[1]);
        }
    }
}

void guardarIndiceDeDanoEnCSV(Cuadricula* cuadricula, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "No se pudo abrir %s para escritura\n", filename);
        return;
    }
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(file, "%.8e", cuadricula->indices_dano[i][j]);
            if (j < GRID_X - 1) fprintf(file, ",");
        }
        fprintf(file, "\n");
    }
    fclose(file);
    printf("Guardado: %s\n", filename);
}

