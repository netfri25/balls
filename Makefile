CC = clang -fcolor-diagnostics
BUILD = build
SRC = src
# CC = gcc
# CFLAGS += -flax-vector-conversions

CFLAGS += -O3 -std=c11 -mtune=native -march=native -Wall -Wextra -I./thirdparty

all: ${BUILD} ${BUILD}/naive ${BUILD}/vect

${BUILD}:
	mkdir -p ${BUILD}

${BUILD}/naive: ${SRC}/main.c ${SRC}/naive.c ${SRC}/common.h
	${CC} ${CFLAGS} ${SRC}/main.c -DNAIVE thirdparty/libraylib.a -lm -o ${BUILD}/naive

${BUILD}/vect: ${SRC}/main.c ${SRC}/vect.c ${SRC}/common.h
	${CC} ${CFLAGS} ${SRC}/main.c thirdparty/libraylib.a -lm -o ${BUILD}/vect

clean:
	rm -rf ${BUILD}
