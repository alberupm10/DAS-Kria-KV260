#include "vitis_fft/hls_ssr_fft_enums.hpp"
#include "vt_fft.hpp"
#include <cmath>
#include <hls_stream.h>
#include <complex>
#include <ap_fixed.h>
#include "hls_math.h"


#define FFT_LENGTH 4096
#define NUM_SIGNALS 132
#define FFT_SSR 4      
#define IID_FFT 0
#define IID_IFFT 1           
#define VELOCIDAD_ONDA 5000 
#define FRECUENCIA_MUESTREO 12500000

// USAR NAMESPACE PARA LLAMAR A LA FUNCION FFT DIRECTAMENTE
using namespace xf::dsp::fft;

//TIPO DE DATO PARA LAS TRANSFORMADAS
typedef ap_fixed<32, 2> fixed_t_fft;
typedef std::complex<fixed_t_fft> T_in;

//CONSTANTES UTILZADAS EN TRIP-COUNT
const int c_length = FFT_LENGTH;
const int c_ssr = FFT_SSR;
const int c_nyquist = FFT_LENGTH/2;
const int c_sizeD_max = 90000; //300x300
const int c_sizeD_min = 5625; //75x75
const int c_num_signals = NUM_SIGNALS;
const int c_fs = FRECUENCIA_MUESTREO;
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
    int j;
    input_rd:
    for (int i = 0; i < size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_length max = c_length
        T_in value;
        j = i % c_ssr; //SELECCIONO EL ARRAY ADECUADO PARA LOS STREAMS DE LA FFT
        value.real(input[i]);
        value.imag(0);
        buffer_input[i]=value.real();
        input_array[j].write(value);
    }
}

// FUNCION PARA CARGAR LAS DISTANCIAS EN LOS STREAMS
void store_distances(float* distancias_host, hls::stream<float>& distances,int size) {
    dist_rd:    
    for (int i = 0; i < size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_sizeD_min max = c_sizeD_max
        distances << distancias_host[i];
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
        j = i % c_ssr;//SELECCIONO EL ARRAY ADECUADO PARA LOS STREAMS DE LA FFT
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
    store_hilbert:
    for (int i = 0; i < size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_length max = c_length
        j = i % c_ssr;//SELECCIONO EL ARRAY ADECUADO PARA LOS STREAMS DE LA FFT
        value = output_array[j].read();
        buffer_hilbert[i]=value.imag();
    }
}

//FUNCION PARA CALCULAR Y ACUMULAR EL INDICE DE DAÑO Y ALMACENARLO DIRECTAMENTE EN MEMORIA
void calculate_index_damage(hls::stream<float>& distances_E, hls::stream<float>& distances_R,
                            float* output, fixed_t_fft buffer_input[c_length],
                            fixed_t_fft buffer_hilbert[c_length], int size_D, int index_signal) {
    float result,d0real,d0imag,D_0_real, D_0_imag, de, dr, realH_ser, imagH_ser, factor,real_temp,imag_temp;
    float t0;
    int n0;
    float Ts = 1.0 / c_fs;
    std::complex<float> value_temp,suma;
    static std::complex<float> index_temp[c_sizeD_max] = {}; // BUFFER PARA ACUMULAR EL ÍNDICE DE DAÑO ENTRE SEÑALES
    #pragma HLS BIND_OP variable=result op=fsqrt impl=fulldsp
    #pragma HLS BIND_OP variable=factor op=fsqrt impl=fulldsp
    #pragma HLS BIND_OP variable=D_0_real op=fmul impl=fulldsp
    #pragma HLS BIND_OP variable=D_0_imag op=fmul impl=fulldsp
    #pragma HLS BIND_OP variable=d0real op=fmul impl=fulldsp
    #pragma HLS BIND_OP variable=d0imag op=fmul impl=fulldsp
    #pragma HLS BIND_STORAGE variable=index_temp type=ram_2p impl=uram 
    calculate:
    for (int i = 0; i < size_D; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=c_sizeD_min max=c_sizeD_max
        // INICIALIZACIÓN DEL ÍNDICE DE DAÑO
        D_0_real = 0.0;
        D_0_imag = 0.0;
        // LEO LAS DISTANCIAS ASOCIADAS A ESTA SEÑAL
        de = distances_E.read();
        dr = distances_R.read();
        // CALCULO LA MUESTRA CORRESPONDIENTE
        t0 = ((de + dr) / VELOCIDAD_ONDA) / Ts;
        n0 = (ceil(t0));
        if (n0 >= 0 && n0 < c_length) {
            realH_ser = buffer_hilbert[n0];
            imagH_ser = buffer_input[n0];

            factor = hls::sqrt(de * dr);
            D_0_real = factor * realH_ser;
            D_0_imag = factor * imagH_ser;
        }
        value_temp.real(D_0_real);
        value_temp.imag(D_0_imag);
        // ACUMULAR EL DAÑO ENTRE SEÑALES
        index_temp[i] += value_temp;
        // ALMACENAR EL RESULTADO FINAL EN MEMORIA SI ES LA ÚLTIMA SEÑAL
        if (index_signal == c_num_signals - 1) {
           d0real = index_temp[i].real();
           d0imag = index_temp[i].imag();
           result = hls::sqrt(d0real * d0real + d0imag * d0imag);
           output[i] = result;
        } 
    }
}
// KERNEL
extern "C" {
void krnl_vdfd(float* input,float* distancias_E,float* distancias_R,float* output,int size_FFT,int size_D,int num_signals) {
    #pragma HLS INTERFACE m_axi port=input bundle=gmem0 offset=slave
    #pragma HLS INTERFACE m_axi port=distancias_E bundle=gmem1 offset=slave
    #pragma HLS INTERFACE m_axi port=distancias_R bundle=gmem2 offset=slave
    #pragma HLS INTERFACE m_axi port=output bundle=gmem3 offset=slave

    #pragma HLS INTERFACE s_axilite port=input bundle=control 
    #pragma HLS INTERFACE s_axilite port=distancias_E bundle=control
    #pragma HLS INTERFACE s_axilite port=distancias_R bundle=control
    #pragma HLS INTERFACE s_axilite port=output bundle=control
    #pragma HLS INTERFACE s_axilite port=size_FFT bundle=control
    #pragma HLS INTERFACE s_axilite port=size_D bundle=control
    #pragma HLS INTERFACE s_axilite port=num_signals bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // HLS STREAM PARA LA ENTRADA Y SALIDA DE FFT (FORWARD)
    hls::stream<T_in> input_array[c_ssr];
    hls::stream<T_FFTout> fft_out[c_ssr];
    // HLS STREAM PARA LA ENTRADA Y SALIDA DE IFFT (REVERSE)
    hls::stream<T_in> ifft_in[c_ssr];
    hls::stream<T_IFFTout> output_array[c_ssr];
    // HLS STREAM PARA LAS DISTANCIAS Y EL INDICE DE DAÑO
    hls::stream<float> distance_E;
    hls::stream<float> distance_R;
    // BUFFERS PARA ALMACENAR LA ENTRADA Y LA TRANSFORMADA HILBERT    
    fixed_t_fft buffer_input[c_length];
    fixed_t_fft buffer_hilbert[c_length];
    // DESPLAZAMIENTOS EN LA ENTRADA (SEÑALES) Y DISTANCIAS
    int block_offset = 0, dist_offset = 0;

    #pragma HLS STREAM variable=input_array depth=16
    #pragma HLS STREAM variable=fft_out depth=16
    #pragma HLS STREAM variable=ifft_in depth=16
    #pragma HLS STREAM variable=output_array depth=16
    #pragma HLS STREAM variable=distance_E depth=16
    #pragma HLS STREAM variable=distance_R depth=16

    #pragma HLS ARRAY_PARTITION variable=input_array complete dim=1
    #pragma HLS ARRAY_PARTITION variable=fft_out complete dim=1
    #pragma HLS ARRAY_PARTITION variable=ifft_in complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output_array complete dim=1

    #pragma HLS BIND_STORAGE variable=buffer_input type=ram_s2p impl=uram 
    #pragma HLS BIND_STORAGE variable=buffer_hilbert type=ram_s2p impl=uram 

    DelayAndSum:
    for(int index_signal = 0;index_signal<num_signals;index_signal++){
        #pragma HLS DATAFLOW
        #pragma HLS LOOP_TRIPCOUNT min=c_num_signals max=c_num_signals   
        block_offset = index_signal*c_length;        
        dist_offset = index_signal*size_D;

        store_distances(distancias_E+dist_offset, distance_E,size_D);

        store_distances(distancias_R+dist_offset, distance_R,size_D);

        load_data_fft(input+block_offset, input_array,buffer_input,size_FFT);

        fft<fft_params, IID_FFT>(input_array, fft_out);

        apply_hilbert(fft_out, ifft_in,size_FFT);     

        fft<ifft_params, IID_IFFT>(ifft_in, output_array);

        store_data_hilbert(output_array, buffer_hilbert, size_FFT);

        calculate_index_damage(distance_E,distance_R,output,buffer_input,buffer_hilbert,size_D,index_signal);
    }
}
}