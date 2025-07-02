CFLAGS += -O3
LDFLAGS += -lfftw3 -lm

TARGET = delay_and_sum_arm64
SRC = delay_and_sum_arm64.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

