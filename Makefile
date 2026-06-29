CC = clang -fcolor-diagnostics
BUILD = build
SRC = src
# CC = gcc
# CFLAGS += -flax-vector-conversions

CFLAGS += -O3 -std=c11 -mtune=native -march=native -Wall -Wextra -I./thirdparty

all: ${BUILD} ${BUILD}/naive

${BUILD}:
	mkdir -p ${BUILD}

${BUILD}/naive: ${SRC}/main.c ${SRC}/naive.c ${SRC}/common.h
	${CC} ${CFLAGS} ${SRC}/main.c thirdparty/libraylib.a -lm -o ${BUILD}/naive

avx2: ${BUILD}/avx2

${BUILD}/avx2: ${SRC}/main.c ${SRC}/avx2.c ${SRC}/common.h
	${CC} ${CFLAGS} ${SRC}/main.c thirdparty/libraylib.a -lm -o ${BUILD}/avx2 -DAVX2

clean:
	rm -rf ${BUILD}
