#pragma once

#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CIRCLE_COUNT (16*600)
#define MIN_RADIUS 2
#define MAX_RADIUS 5


struct Sampler {
    size_t count;
    double avg;
    double highest;
    double lowest;
};

#define MEASURE(sampler, block) \
    do { \
        clock_t const start = clock(); \
        block; \
        clock_t const elapsed = clock() - start; \
        double const update_time_ms = (double) elapsed / (double) CLOCKS_PER_SEC * 1000; \
        (sampler)->avg = ((sampler)->avg * (sampler)->count + update_time_ms) / ((sampler)->count + 1); \
        (sampler)->count++; \
        (sampler)->highest = fmaxf((sampler)->highest, update_time_ms); \
        if ((sampler)->lowest == 0) (sampler)->lowest = INFINITY; \
        (sampler)->lowest = fminf((sampler)->lowest, update_time_ms); \
    } while (0)

#define da_reserve_additional(da, expected_additional_capacity) \
    do { \
        size_t const expected_capacity = (da)->count + (expected_additional_capacity); \
        if ((expected_capacity) > (da)->capacity) { \
            if ((da)->capacity == 0) { \
                (da)->capacity = 4; \
            } \
            while ((expected_capacity) > (da)->capacity) { \
                (da)->capacity *= 2; \
            } \
            (da)->items = realloc((da)->items, (da)->capacity * sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol"); \
        } \
    } while (0)

#define da_append(da, item) \
    do { \
        da_reserve_additional((da), 1); \
        (da)->items[(da)->count++] = (item); \
    } while (0)

#define da_free(da) \
    do { \
        if ((da)->items) free((da)->items); \
    } while (0);


static float rand_float(float low, float high) {
    return (float) rand() / (double) RAND_MAX * (high - low) + low;
}

