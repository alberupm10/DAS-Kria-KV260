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
#include <chrono>

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

#define FFT_LENGTH 4096// Tamaño de la FFT
#define NUM_SIGNALS 132

#define GRID_X 120
#define GRID_Y 120
#define NUM_SENSORES 12
#define VELOCIDAD_ONDA 5000 // Velocidad de la onda mecánica en m/s

#define OUTPUT_FILE "output_vdm.csv"
#define CONFIG_FILE "config_vdm.txt"
//FUNCIONES
void producer(const std::string& input_file);
void consumer(float* organized_data);
void calcularDistanciasHost(float* distancias_host);
void extract_sensor_distances(float* distancias_E, float* distancias_R, 
                              int E, int R);
void write_index_damage_to_file(float* index_damage_result);
void write_config_to_file(const std::string& filename, size_t bytes_input, size_t bytes_distances, 
                          size_t bytes_output, double media_tiempo, double desviacion_tiempo);
double calcularMedia(const std::vector<double>& valores);
double calcularDesviacion(const std::vector<double>& valores, double media);
void calcularDistanciasEnHilo(float* distancias_host);
void consumerBuffer(float* organized_data);

static const int DATA_SIZE = FFT_LENGTH;
static const int TOTAL_SIZE = FFT_LENGTH*NUM_SIGNALS;
static const int INDEX_DAMAGE_SIZE = GRID_X*GRID_Y;
static const int DISTANCES_SIZE = GRID_X*GRID_Y*NUM_SENSORES;
static const int num_signal = NUM_SIGNALS;

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

// Posiciones de los sensores
float sensores_x[NUM_SENSORES] = {0, 0.009, 0.018, 0.027, 0.036, 0.045, 0.054, 0.063, 0.072, 0.081, 0.090, 0.099};
float sensores_y[NUM_SENSORES] = {0};

// Buffer compartido
std::queue<std::vector<float>> buffer;
std::mutex buffer_mutex;
std::condition_variable buffer_cv;
bool done = false;
// Variable global para almacenar datos organizados
std::vector<float> organized_data(TOTAL_SIZE);


int main(int argc, char* argv[]) {
    // TARGET_DEVICE macro needs to be passed from gcc command line
    if (argc != 8) {
        std::cout << "Usage: " << argv[0] 
                << " <xclbin> <inputFile> <X_START> <X_END> <Y_START> <Y_END> <N_iter>"
                << std::endl;
        return EXIT_FAILURE;
    }

    auto start_time_total = std::chrono::high_resolution_clock::now();

    std::string xclbinFilename = argv[1];
    std::string inputFile = argv[2];

    X_START = std::atof(argv[3]);
    X_END = std::atof(argv[4]);
    Y_START = std::atof(argv[5]);
    Y_END = std::atof(argv[6]);

    N_iter = std::atoi(argv[7]);
    //Calcular automáticamente el ancho y alto en función de los límites
    ANCHO = X_END - X_START;
    ALTO = Y_END - Y_START;
    // Compute the size of array in bytes
    size_t size_in_bytes_input = TOTAL_SIZE * sizeof(float); // Datos complejos
    size_t size_in_bytes_distances = DISTANCES_SIZE * sizeof(float); // Datos complejos
    size_t size_in_bytes_output = INDEX_DAMAGE_SIZE * sizeof(float); // Datos complejos

    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    // using customized allocator for getting buffer alignment to 4k boundary

    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl_vdm;
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
            OCL_CHECK(err, krnl_vdm = cl::Kernel(program, "krnl_vdm", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    float* ptr_input;
    float* ptr_result;  
    float* distancias_host;

    ptr_input = new float[TOTAL_SIZE]; 
    distancias_host = new float[GRID_X * GRID_Y * NUM_SIGNALS];

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, size_in_bytes_input, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dist(context, CL_MEM_READ_ONLY, size_in_bytes_distances, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_result(context, CL_MEM_WRITE_ONLY, size_in_bytes_output, NULL, &err));

    // set the kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_vdm.setArg(narg++, buffer_input));
    OCL_CHECK(err, err = krnl_vdm.setArg(narg++, buffer_dist));
    OCL_CHECK(err, err = krnl_vdm.setArg(narg++, buffer_result));
    OCL_CHECK(err, err = krnl_vdm.setArg(narg++, DATA_SIZE));
    OCL_CHECK(err, err = krnl_vdm.setArg(narg++, num_signal));

    // We then need to map our OpenCL buffers to get the pointers
    std::vector<double> tiempos_ejecucion;
   
    OCL_CHECK(err,
              ptr_input = (float*)q.enqueueMapBuffer(buffer_input, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_input, NULL, NULL, &err));
    OCL_CHECK(err,
              distancias_host = (float*)q.enqueueMapBuffer(buffer_dist, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_distances, NULL, NULL, &err));

    OCL_CHECK(err, ptr_result = (float*)q.enqueueMapBuffer(buffer_result, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_output, NULL,
                                                         NULL, &err));

    std::thread producer_thread(producer, inputFile);
    std::thread consumer_thread(consumer,ptr_input);
    std::thread distancias_thread(calcularDistanciasEnHilo,distancias_host);

    producer_thread.join();
    consumer_thread.join();
    distancias_thread.join();

    // Data will be migrated to kernel space*/
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input,buffer_dist}, 0 /* 0 means from host*/));

    // **Ejecutar el kernel N veces**
    for (int i = 0; i < N_iter; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        cl::Event event;
    
        OCL_CHECK(err, q.enqueueTask(krnl_vdm, nullptr, &event));
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
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dist, distancias_host));
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

        for (const auto& value : data) {
            if (global_index < TOTAL_SIZE) {
                organized_data[global_index++] = value;
            }
        }
    }
    std::cout << "Consumer: Organización de datos completa.\n";
    std::cout << "global_index: " << global_index << "\n";
}

void calcularDistanciasHost(float* distancias_host) {
    for (int q = 0; q < GRID_Y; q++) {
        for (int w = 0; w < GRID_X; w++) {
            double x = X_START + w * ((ANCHO) / (GRID_X - 1));
            double y = Y_START + q * ((ALTO) / (GRID_Y - 1));

            for (int s = 0; s < NUM_SENSORES; s++) {
                double dx = x - sensores_x[s];
                double dy = y - sensores_y[s];

                // Convertimos el índice 3D a un índice 1D
                int idx = (q * GRID_X * NUM_SENSORES) + (w * NUM_SENSORES) + s; 
                
                distancias_host[idx] = sqrt(dx * dx + dy * dy);
            }
        }
    }
}
void calcularDistanciasEnHilo(float* distancias_host) {
    auto start_time_dist = std::chrono::high_resolution_clock::now();
    calcularDistanciasHost(distancias_host);
    auto end_time_dist = std::chrono::high_resolution_clock::now();
    tiempoDist = std::chrono::duration<double>(end_time_dist - start_time_dist).count();
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

// Función para guardar la configuración en un archivo
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