#tar -xvzf fftw-3.3.10.tar.gz

source ../env/environment-setup-cortexa72-cortexa53-xilinx-linux
cd fftw-3.3.10

./configure \
    --host=aarch64-linux \
    --prefix=$SDKTARGETSYSROOT/usr \
    --enable-shared \
    --enable-static

make -j$(nproc)
make install
