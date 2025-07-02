CC = gcc
CFLAGS = -O3
LIBS = -lfftw3 -lm

TARGET = delay_and_sum_pc
SRC = delay_and_sum_pc.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)

