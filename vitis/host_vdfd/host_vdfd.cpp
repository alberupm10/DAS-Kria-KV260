#include "common.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <complex>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <ap_fixed.h>
#include <complex>
#include <cmath>
#include <chrono>

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

#define FFT_LENGTH 4096// Tamaño de la FFT
#define NUM_SIGNALS 132

#define NUM_SENSORES 12
#define VELOCIDAD_ONDA 5000 // Velocidad de la onda mecánica en m/s
#define OUTPUT_FILE "output_vdfd.csv"
#define CONFIG_FILE "config_vdfd.txt"

//FUNCIONES
void producer(const std::string& input_file);
void consumer(float* organized_data);
void calcularDistanciasHost();
void extract_sensor_distances(float* distancias_E, float* distancias_R, 
                              int E, int R);
void write_index_damage_to_file(float* index_damage_result);
void get_emisor_receptor(int index_signal, int& E, int& R) ;
void calcularDistanciasHostStream(float* distancias_E_host, float* distancias_R_host);
void write_config_to_file(const std::string& filename, size_t bytes_input, size_t bytes_distances, 
                          size_t bytes_output, double media_tiempo, double desviacion_tiempo);
double calcularMedia(const std::vector<double>& valores);
double calcularDesviacion(const std::vector<double>& valores, double media);
void calcularDistanciasEnHilo(float* distancias_E_host, float* distancias_R_host);
void consumerBuffer(float* organized_data);
//VALORES CONFIGURABLES
int GRID_X = 200;  // VALOR POR DEFECTO
int GRID_Y = 200;  // VALOR POR DEFECTO

float X_START = -0.3;
float X_END = 0.5;
float Y_START = -0.05;
float Y_END = 0.65;

float ANCHO;
float ALTO;

int N_iter = 1;
double tiempoDist;
double tiempokrnl;
double tiempoTotal;
static const int DATA_SIZE = FFT_LENGTH;

static const int TOTAL_SIZE = FFT_LENGTH*NUM_SIGNALS;
static const int NUM_SIGNALS_SIZE = NUM_SIGNALS;

// Posiciones de los sensores
float sensores_x[NUM_SENSORES] = {0, 0.009, 0.018, 0.027, 0.036, 0.045, 0.054, 0.063, 0.072, 0.081, 0.090, 0.099};
float sensores_y[NUM_SENSORES] = {0};

// Buffer compartido
std::queue<std::vector<float>> buffer;
std::mutex buffer_mutex;
std::condition_variable buffer_cv;
bool done = false;

// Implementación básica de FFT y su IFFT CON EL METODO OVERLAP SAVE
int main(int argc, char* argv[]) {
    // TARGET_DEVICE macro needs to be passed from gcc command line
    if (argc != 10) {
        std::cout << "Usage: " << argv[0] 
                << " <xclbin> <inputFile> <GRID_X> <GRID_Y> <X_START> <X_END> <Y_START> <Y_END> <N_iter>"
                << std::endl;
        return EXIT_FAILURE;
    }

    auto start_time_total = std::chrono::high_resolution_clock::now();

    std::string xclbinFilename = argv[1];
    std::string inputFile = argv[2];

    GRID_X = std::atoi(argv[3]);
    GRID_Y = std::atoi(argv[4]);

    X_START = std::atof(argv[5]);
    X_END = std::atof(argv[6]);
    Y_START = std::atof(argv[7]);
    Y_END = std::atof(argv[8]);

    N_iter = std::atoi(argv[9]);

    ANCHO = X_END - X_START;
    ALTO = Y_END - Y_START;

    int INDEX_DAMAGE_SIZE = GRID_X * GRID_Y;
    size_t size_in_bytes_input = TOTAL_SIZE * sizeof(float); 
    size_t size_in_bytes_distances = INDEX_DAMAGE_SIZE * NUM_SIGNALS * sizeof(float);
    size_t size_in_bytes_output = INDEX_DAMAGE_SIZE * sizeof(float);

    std::vector<double> tiempos_ejecucion;

    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl_vdfd;
    cl::Program program;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    // traversing all Platforms To find Xilinx Platform and targeted
    // Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx") {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size()) {
                found_device = true;
                break;
            }
        }
    }
    if (found_device == false) {
        std::cout << "Error: Unable to find Target Device " << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "INFO: Reading " << xclbinFilename << std::endl;
    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_vdfd = cl::Kernel(program, "krnl_vdfd", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    float* ptr_input;
    float* ptr_dist_E;
    float* ptr_dist_R;
    float* ptr_result;
    
    ptr_input = new float[TOTAL_SIZE];  
    ptr_dist_E = new float[GRID_X * GRID_Y * NUM_SIGNALS]; 
    ptr_dist_R = new float[GRID_X * GRID_Y * NUM_SIGNALS]; 

    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, size_in_bytes_input, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dist_E(context, CL_MEM_READ_ONLY, size_in_bytes_distances, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dist_R(context, CL_MEM_READ_ONLY, size_in_bytes_distances, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_result(context, CL_MEM_WRITE_ONLY, size_in_bytes_output, NULL, &err));

    // set the kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_vdfd.setArg(narg++, buffer_input));
    OCL_CHECK(err, err = krnl_vdfd.setArg(narg++, buffer_dist_E));
    OCL_CHECK(err, err = krnl_vdfd.setArg(narg++, buffer_dist_R));
    OCL_CHECK(err, err = krnl_vdfd.setArg(narg++, buffer_result));
    OCL_CHECK(err, err = krnl_vdfd.setArg(narg++, DATA_SIZE));
    OCL_CHECK(err, err = krnl_vdfd.setArg(narg++, INDEX_DAMAGE_SIZE));
    OCL_CHECK(err, err = krnl_vdfd.setArg(narg++, NUM_SIGNALS_SIZE));

    // We then need to map our OpenCL buffers to get the pointers
    OCL_CHECK(err,
              ptr_input = (float*)q.enqueueMapBuffer(buffer_input, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_input, NULL, NULL, &err));
    OCL_CHECK(err,
              ptr_dist_E = (float*)q.enqueueMapBuffer(buffer_dist_E, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_distances, NULL, NULL, &err));
    OCL_CHECK(err,
              ptr_dist_R = (float*)q.enqueueMapBuffer(buffer_dist_R, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_distances, NULL, NULL, &err));

    OCL_CHECK(err, ptr_result = (float*)q.enqueueMapBuffer(buffer_result, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_output, NULL,
                                                         NULL, &err));

// **Lanzar hilos de productor y consumidor**
    std::thread producer_thread(producer, inputFile);
    std::thread consumer_thread(consumer,ptr_input);
    std::thread distancias_thread(calcularDistanciasEnHilo,ptr_dist_E,ptr_dist_R);
    // **Esperar a que terminen ambos hilos**
    producer_thread.join();
    consumer_thread.join();
    distancias_thread.join();
    //consumerBuffer(ptr_input);
    
    // Data will be migrated to kernel space*/     
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input,buffer_dist_E,buffer_dist_R}, 0 /* 0 means from host*/));

    // **Ejecutar el kernel N veces**
    for (int i = 0; i < N_iter; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        cl::Event event;
    
        OCL_CHECK(err, q.enqueueTask(krnl_vdfd, nullptr, &event));
        event.wait();
        auto end_time = std::chrono::high_resolution_clock::now();
        tiempokrnl = std::chrono::duration<double>(end_time - start_time).count();
        tiempos_ejecucion.push_back(tiempokrnl);
    }
    
    double media = calcularMedia(tiempos_ejecucion);
    double desviacion = calcularDesviacion(tiempos_ejecucion, media);

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_input, ptr_input));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dist_E, ptr_dist_E));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dist_R, ptr_dist_R));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_result, ptr_result));
    OCL_CHECK(err, err = q.finish());
    
    auto end_time_total = std::chrono::high_resolution_clock::now();
    tiempoTotal = std::chrono::duration<double>(end_time_total - start_time_total).count();
    write_index_damage_to_file(ptr_result);
    write_config_to_file(CONFIG_FILE, size_in_bytes_input, size_in_bytes_distances, size_in_bytes_output, media, desviacion);

    return 0;

}

// Hilo productor
void producer(const std::string& input_file) {
    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "Error al abrir el archivo de entrada.\n";
        return;
    }

    std::cout << "Producer: Inicio de lectura del archivo.\n";

    int line_count = 0;
    std::string line;

    while (line_count < NUM_SIGNALS && std::getline(infile, line)) {
        std::vector<float> data(FFT_LENGTH); // Contenedor para una línea completa
        std::stringstream ss(line);
        float value;
        int index = 0;

        while (ss >> value) {
            data[index++] = value;
            if (ss.peek() == ',') ss.ignore();
        }

        if (index != FFT_LENGTH) {
            std::cerr << "Error: Línea " << line_count << " no contiene exactamente " << FFT_LENGTH << " valores.\n";
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(buffer_mutex);
            buffer_cv.wait(lock, [] { return buffer.size() < NUM_SIGNALS; });
            buffer.push(data);
            //std::cout << "Producer: Línea " << line_count << " almacenada en el buffer.\n";
            //std::cout << "Producer: Primer dato en línea " << line_count << ": " << data[24000] << ".\n";
            line_count++;
        }

        buffer_cv.notify_all();
    }

    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        done = true;
    }
    buffer_cv.notify_all();
    infile.close();
    std::cout << "Producer: Finalizado. Líneas leídas: " << line_count << "\n";
}


// Hilo consumidor
void consumer(float* organized_data) {
    int global_index = 0;
    while (true) {
        std::vector<float> data;
        {
            std::unique_lock<std::mutex> lock(buffer_mutex);
            buffer_cv.wait(lock, [] { return !buffer.empty() || done; });

            if (buffer.empty() && done) break;          
            data = buffer.front();
            buffer.pop();
        }
        buffer_cv.notify_all();

        // Organizar los datos en la estructura global
        for (const auto& value : data) {
            if (global_index < TOTAL_SIZE) {
                organized_data[global_index++] = value;
            }
        }
    }
    std::cout << "Consumer: Organización de datos completa.\n";
    std::cout << "global_index: " << global_index << "\n";
}

void calcularDistanciasHostStream(float* distancias_E_host, float* distancias_R_host) {
    for (int q = 0; q < GRID_Y; q++) {
        for (int w = 0; w < GRID_X; w++) {
            float x = X_START + w * ((ANCHO) / (GRID_X - 1));
            float y = Y_START + q * ((ALTO) / (GRID_Y - 1));

            for (int signal_idx = 0; signal_idx < NUM_SIGNALS; signal_idx++) {
                int E, R;
                get_emisor_receptor(signal_idx, E, R);

                float dx_E = x - sensores_x[E - 1];
                float dy_E = y - sensores_y[E - 1];

                float dx_R = x - sensores_x[R - 1];
                float dy_R = y - sensores_y[R - 1];

                int index = q * GRID_X + w + signal_idx * (GRID_X * GRID_Y);

                distancias_E_host[index] = sqrt(dx_E * dx_E + dy_E * dy_E);
                distancias_R_host[index] = sqrt(dx_R * dx_R + dy_R * dy_R);
            }
        }
    }
}


void write_index_damage_to_file(float* index_damage_result) {
    std::ofstream outfile(OUTPUT_FILE);
    if (!outfile.is_open()) {
        std::cerr << "Error al abrir el archivo de salida: " << OUTPUT_FILE << "\n";
        return;
    }

    for (int q = 0; q < GRID_Y; q++) {
        for (int w = 0; w < GRID_X; w++) {
            int idx = q * GRID_X + w;
            outfile << std::scientific << std::setprecision(8) << index_damage_result[idx];

            if (w < GRID_X - 1) {
                outfile << ",";
            }
        }
        outfile << "\n";
    }

    outfile.close();
    std::cout << "Índice de daño guardado en formato CSV en " << OUTPUT_FILE << "\n";
}

void get_emisor_receptor(int index_signal, int& E, int& R) {
    int count = 0;
    for (int e = 1; e <= NUM_SENSORES; e++) {
        for (int r = 1; r <= NUM_SENSORES; r++) {
            if (e != r) { 
                if (count == index_signal) {
                    E = e;
                    R = r;
                    return;
                }
                count++;
            }
        }
    }
}


void write_config_to_file(const std::string& filename, size_t bytes_input, size_t bytes_distances, 
                          size_t bytes_output, double media_tiempo, double desviacion_tiempo) {
    std::ofstream config_file(filename);
    if (!config_file.is_open()) {
        std::cerr << "Error al abrir el archivo de configuración: " << filename << "\n";
        return;
    }

    config_file << "GRID_X=" << GRID_X << "\n";
    config_file << "GRID_Y=" << GRID_Y << "\n";
    config_file << "X_START=" << X_START << "\n";
    config_file << "X_END=" << X_END << "\n";
    config_file << "Y_START=" << Y_START << "\n";
    config_file << "Y_END=" << Y_END << "\n";
    config_file << "ANCHO=" << ANCHO << "\n";
    config_file << "ALTO=" << ALTO << "\n";
    
    config_file << "BYTES_INPUT=" << (bytes_input / (1024.0 * 1024.0)) << " MB\n";
    config_file << "BYTES_DISTANCES=" << (bytes_distances / (1024.0 * 1024.0)) << " MB\n";
    config_file << "BYTES_OUTPUT=" << (bytes_output / (1024.0 * 1024.0)) << " MB\n";

    config_file << "MEDIA_TIEMPO_KERNEL=" << (media_tiempo * 1000.0) << " ms\n";
    config_file << "DESVIACION_TIEMPO_KERNEL=" << (desviacion_tiempo * 1000.0) << " ms\n";
    config_file << "TIEMPO CALCULO DISTANCIAS=" << tiempoDist << "s\n";
    config_file << "TIEMPO TOTAL=" << tiempoTotal << "s\n";

    config_file << "N=" << N_iter <<"\n";
    config_file.close();
    std::cout << "Configuración guardada en " << filename << "\n";
}

double calcularMedia(const std::vector<double>& valores) {
    double suma = 0.0;
    for (double v : valores) suma += v;
    return suma / valores.size();
}

double calcularDesviacion(const std::vector<double>& valores, double media) {
    double suma = 0.0;
    for (double v : valores) suma += (v - media) * (v - media);
    return sqrt(suma / valores.size());
}

void calcularDistanciasEnHilo(float* distancias_E_host, float* distancias_R_host) {
    auto start_time_dist = std::chrono::high_resolution_clock::now();
    calcularDistanciasHostStream(distancias_E_host, distancias_R_host);
    auto end_time_dist = std::chrono::high_resolution_clock::now();
    tiempoDist = std::chrono::duration<double>(end_time_dist - start_time_dist).count();
}

void consumerBuffer(float* organized_data) {
    int global_index = 0;
    while (true) {
        std::vector<float> data;
        {
            std::unique_lock<std::mutex> lock(buffer_mutex);
            buffer_cv.wait(lock, [] { return !buffer.empty() || done; });

            if (buffer.empty() && done) break;
            data = buffer.front();
            buffer.pop();
        }
        buffer_cv.notify_all();

        for (const auto& value : data) {
            if (global_index < TOTAL_SIZE) {
                organized_data[global_index++] = value;
            }
        }
    }
    std::cout << "Datos organizados por el consumido correctamente.\n";
}
