#include "vitis_fft/hls_ssr_fft_enums.hpp"
#include "vt_fft.hpp"
#include <hls_stream.h>
#include <complex>
#include <ap_fixed.h>
#include "hls_math.h"

#define FFT_LENGTH 4096// Tamaño de la FFT
#define FFT_SSR 4      
#define IID_FFT 0
#define IID_IFFT 1           
#define NUM_SENSORES 12
#define NUM_SIGNALS 132
#define GRID_X 120
#define GRID_Y 120
#define VELOCIDAD_ONDA 5000 // Velocidad de la onda mecánica en m/s
#define FRECUENCIA_MUESTREO 12.5E6

// USAR NAMESPACE PARA LLAMAR A LA FUNCION FFT DIRECTAMENTE
using namespace xf::dsp::fft;

//TIPO DE DATO PARA LAS TRANSFORMADAS
typedef ap_fixed<32, 2> fixed_t_fft;
typedef std::complex<fixed_t_fft> T_in;

//CONSTANTES UTILZADAS EN TRIP-COUNT
const int c_length = FFT_LENGTH;
const int c_ssr = FFT_SSR;
const int c_nyquist = FFT_LENGTH/2;
const int c_distances = NUM_SENSORES*GRID_X*GRID_Y;
const int c_index_damage = GRID_X*GRID_Y;
const int c_grid_x = GRID_X;
const int c_grid_y = GRID_Y;
const int c_num_sensors = NUM_SENSORES;
const int c_num_sensors_loop = NUM_SENSORES-1;
const int c_num_signals = NUM_SIGNALS;

// PARAMETROS DE CONFIGURACION DE LA FFT (FORWARD)
struct fft_params : ssr_fft_default_params {
    static const int N = FFT_LENGTH;                        
    static const int R = FFT_SSR;                           
    static const scaling_mode_enum scaling_mode = SSR_FFT_SCALE; 
    static const fft_output_order_enum output_data_order = SSR_FFT_NATURAL;
    static const transform_direction_enum transform_direction = FORWARD_TRANSFORM;  
    static const butterfly_rnd_mode_enum butterfly_rnd_mode = TRN;
    static const int twiddle_table_word_length = 32;
    static const int twiddle_table_intger_part_length = 2; 
};

//TIPO DE DATO PARA LA SALIDA FFT(FORWARD) 
typedef ssr_fft_output_type<fft_params, T_in>::t_ssr_fft_out T_FFTout; 

// PARAMETROS DE CONFIGURACION DE LA FFT (REVERSE)
struct ifft_params : ssr_fft_default_params {
    static const int N = FFT_LENGTH;                        
    static const int R = FFT_SSR;                           
    static const scaling_mode_enum scaling_mode = SSR_FFT_SCALE; 
    static const fft_output_order_enum output_data_order = SSR_FFT_NATURAL;
    static const transform_direction_enum transform_direction = REVERSE_TRANSFORM;  
    static const butterfly_rnd_mode_enum butterfly_rnd_mode = TRN;
    static const int twiddle_table_word_length = 32;
    static const int twiddle_table_intger_part_length = 2; 
};
//TIPO DE DATO PARA LA SALIDA FFT(REVERSE) 
typedef ssr_fft_output_type<ifft_params, T_in>::t_ssr_fft_out T_IFFTout; 


// FUNCION PARA CARGAR LAS SEÑALES EN EL STREAM PARA LA FFT Y EN UN BUFFER PARA EL CALCULO DEL DAÑO
void load_data_fft(float* input, hls::stream<T_in> input_array[c_ssr],fixed_t_fft buffer_input[c_length],int size) {
    T_in value;
    int j,index;
    input_rd:
    for (int i = 0; i < size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_length max = c_length    
        j = i % c_ssr;
        value.real(input[i]);
        value.imag(0);
        buffer_input[i]=value.real();
        input_array[j].write(value);
    }
}

void load_distances(float* distances, float* distancias_local) {
    dist_rd:
    for (int i = 0; i < c_distances; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_distances max=c_distances
        distancias_local[i]=distances[i];
    }

}


//FUNCION PARA APLICAR LA TRANSOFORMADA HILBERT
void apply_hilbert(hls::stream<T_FFTout> fft_out[c_ssr], hls::stream<T_in> ifft_in[c_ssr],int size) {
    T_FFTout value;
    T_in converted;
    int j;
    filter:
    for (int i = 0; i < size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_length max = c_length
        j = i % c_ssr;
        value = fft_out[j].read();

        if ((i == 0) || (i == c_nyquist)) {  // DC y frecuencia de Nyquist
            converted.real(0);
            converted.imag(0);
        } else if (i < c_nyquist) {  // Frecuencias positivas
            converted.real(-value.real());
            converted.imag(-value.imag());
        } else {  // Frecuencias negativas
            converted.real(value.real());
            converted.imag(value.imag());
        }
        ifft_in[j].write(converted);
    }
}

//FUNCION PARA ALMACENAR LA TRANSFORMADA HILBERT EN UN BUFFER
void store_data_hilbert(hls::stream<T_IFFTout> output_array[c_ssr],fixed_t_fft buffer_hilbert[c_length],int size){
    T_IFFTout value;
    int j;
    store_filter:
    for (int i = 0; i < size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_length max = c_length
        j = i % c_ssr;
        value = output_array[j].read();
        buffer_hilbert[i]=value.imag();
    }
}


//FUNCION PARA CALCULAR EL INDICE DE DAÑO ASOCIADO A CADA PUNTO DE LA PARTICULA PARA UNA SEÑAL
void calculate_index_damage(float local_distances[c_distances],
                            float* output,
                            fixed_t_fft buffer_input[c_length],
                            fixed_t_fft buffer_hilbert[c_length],int E,int R,int index_signal) {
    float D_0_real, D_0_imag, de, dr, realH_ser, imagH_ser, factor,d0real,d0imag,result;
    float t0;
    int n0,idxE,idxR,i = 0;
    float Ts = float(1.0 / FRECUENCIA_MUESTREO);
    std::complex<float> value_temp;
    static std::complex<float> index_temp[c_index_damage];
    #pragma HLS BIND_OP variable=result op=fsqrt impl=fulldsp
    #pragma HLS BIND_OP variable=factor op=fsqrt impl=fulldsp
    #pragma HLS BIND_OP variable=D_0_real op=fmul impl=fulldsp
    #pragma HLS BIND_OP variable=D_0_imag op=fmul impl=fulldsp
    #pragma HLS BIND_STORAGE variable= index_temp type=ram_2p impl=uram 
    compute:
    for (int q = 0; q < GRID_Y; q++) {
        #pragma HLS LOOP_TRIPCOUNT min=c_grid_y max=c_grid_y
        for (int w = 0; w < GRID_X; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=c_grid_x max=c_grid_x
            D_0_real = 0.0;
            D_0_imag = 0.0;

            idxE = (q * GRID_X + w) * NUM_SENSORES + (E - 1);
            idxR = (q * GRID_X + w) * NUM_SENSORES + (R - 1);

            de = local_distances[idxE];
            dr = local_distances[idxR];
            t0 = ((de + dr) / VELOCIDAD_ONDA) / Ts;
            n0 = t0;

            if (n0 >= 0 && n0 < c_length) {
                realH_ser = buffer_hilbert[n0];
                imagH_ser = buffer_input[n0];

                factor = hls::sqrt(de * dr);
                D_0_real = factor * realH_ser;
                D_0_imag = factor * imagH_ser;
            }
            value_temp.real(D_0_real);
            value_temp.imag(D_0_imag);
            index_temp[i] += value_temp;

            if(index_signal==c_num_signals-1){
                d0real = index_temp[i].real();
                d0imag = index_temp[i].imag();
                result = hls::sqrt(d0real * d0real + d0imag * d0imag);
                output[i] = result;

            }
            i++;
        }
    }
}


void precalculate_emisor_receptor(int emisor_receptor_map[NUM_SIGNALS * 2]) {
    int count = 0;
    for (int e = 1; e <= c_num_sensors; e++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_num_sensors max = c_num_sensors
        for (int r = 1; r <= c_num_sensors; r++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min = c_num_sensors max = c_num_sensors
            if (e != r) {
                if (count < NUM_SIGNALS) {
                    int idx = count * 2; 
                    emisor_receptor_map[idx] = e;      
                    emisor_receptor_map[idx + 1] = r; 
                    count++;
                }
            }
        }
    }
}

void get_emisor_receptor(int index_signal, int& E, int& R, int emisor_receptor_map[NUM_SIGNALS * 2]) {
    int idx = index_signal * 2; 
    E = emisor_receptor_map[idx];      
    R = emisor_receptor_map[idx + 1];  
}

void compute_emisor_receptor(int index_signal, int& E, int& R) {
    int count = 0;
    for (int e = 1; e < c_num_sensors+1; e++) {
        for (int r = 1; r < c_num_sensors+1; r++) {
            #pragma HLS LOOP_TRIPCOUNT min = 2 max = 144
            if (e != r) {
                if (count == index_signal) {
                    E = e;
                    R = r;
                    //break;
                }
                count++;
            }
        }
    }
}

// KERNEL
extern "C" {
void krnl_vdm(float* input,float* distancias,float* output,int size_FFT,int num_signals) {
    #pragma HLS INTERFACE m_axi port=input bundle=gmem0  offset=slave 
    #pragma HLS INTERFACE mode=m_axi port=distancias bundle=gmem1 max_widen_bitwidth=32 offset=slave
    #pragma HLS INTERFACE m_axi port=output bundle=gmem2  offset=slave

    #pragma HLS INTERFACE s_axilite port=input bundle=control
    #pragma HLS INTERFACE s_axilite port=distancias bundle=control
    #pragma HLS INTERFACE s_axilite port=output bundle=control
    #pragma HLS INTERFACE s_axilite port=size_FFT bundle=control
    #pragma HLS INTERFACE s_axilite port=num_signals bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    // HLS STREAM PARA LA ENTRADA Y SALIDA DE FFT (FORWARD)
    static hls::stream<T_in> input_array[c_ssr];
    static hls::stream<T_FFTout> fft_out[c_ssr];
    // HLS STREAM PARA LA ENTRADA Y SALIDA DE IFFT (REVERSE)
    static hls::stream<T_in> ifft_in[c_ssr];
    static hls::stream<T_IFFTout> output_array[c_ssr];
    // BUFFERS PARA ALMACENAR LA ENTRADA,DISTANCIAS, LA TRANSFORMADA HILBERT, INDICE DE DAÑO TEMPORAL Y ACUMULADO    
    fixed_t_fft buffer_input[c_length];
    fixed_t_fft buffer_hilbert[c_length];
    float local_distances[c_distances];
    int emisor_receptor_map[c_num_signals*2];
    int E, R,index_signal = 0,block_offset = 0;

    #pragma HLS STREAM variable=input_array depth=16
    #pragma HLS STREAM variable=fft_out depth=16
    #pragma HLS STREAM variable=ifft_in depth=16
    #pragma HLS STREAM variable=output_array depth=16

    #pragma HLS ARRAY_PARTITION variable=input_array complete dim=1
    #pragma HLS ARRAY_PARTITION variable=fft_out complete dim=1
    #pragma HLS ARRAY_PARTITION variable=ifft_in complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output_array complete dim=1
    #pragma HLS ARRAY_PARTITION variable=emisor_receptor_map complete dim=1

    #pragma HLS BIND_STORAGE variable=buffer_input type=ram_2p impl=uram 
    #pragma HLS BIND_STORAGE variable=buffer_hilbert type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=local_distances type=ram_2p impl=uram

    //#pragma HLS DATAFLOW
    load_distances(distancias, local_distances);

    precalculate_emisor_receptor(emisor_receptor_map);

    for(int index_signal = 0;index_signal < num_signals;index_signal++){
        #pragma HLS DATAFLOW
        #pragma HLS LOOP_TRIPCOUNT min = c_num_signals max = c_num_signals
        block_offset=index_signal*c_length;

        load_data_fft(input+block_offset, input_array, buffer_input, size_FFT);

        fft<fft_params, IID_FFT>(input_array, fft_out);

        apply_hilbert(fft_out, ifft_in, size_FFT);

        fft<ifft_params, IID_IFFT>(ifft_in, output_array);

        store_data_hilbert(output_array, buffer_hilbert, size_FFT);

        get_emisor_receptor(index_signal, E, R, emisor_receptor_map);

        calculate_index_damage(local_distances, output, buffer_input, buffer_hilbert, E, R,index_signal);

    }
}
}