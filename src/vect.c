#include "raylib.h"

#include "common.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

struct State {
    struct Sampler sampler;
    uint32_t len;
    float* restrict r;
    float* restrict px;
    float* restrict py;
    float* restrict vx;
    float* restrict vy;
};

struct IndexPairVec {
    size_t count;
    size_t capacity;

    uint32_t *is;
    uint32_t *js;
};

static void* aligned_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment) {
    if (malloc_usable_size(ptr) >= new_size) {
        return ptr;
    }

    void* new_ptr = aligned_alloc(alignment, new_size);
    if (new_ptr == NULL) return NULL;

    if (ptr != NULL) {
        memcpy(new_ptr, ptr, old_size);
        free(ptr);
    }

    return new_ptr;
}

static void IndexPairVec_reserve_additional(struct IndexPairVec* self, size_t expected_additional_capacity) {
    size_t const expected_capacity = self->count + expected_additional_capacity;

    if (expected_capacity <= self->capacity) {
        return;
    }

    if (self->capacity == 0) {
        self->capacity = 4;
    }

    while (expected_capacity > self->capacity) {
        self->capacity *= 2;
    }

    self->is = aligned_realloc(
        self->is,
        self->count    * sizeof *self->is,
        self->capacity * sizeof *self->is,
        8 * sizeof *self->is
    );

    self->js = aligned_realloc(
        self->js,
        self->count    * sizeof *self->js,
        self->capacity * sizeof *self->js,
        8 * sizeof *self->js
    );

    assert(self->is != NULL && self->js != NULL && ":(");
}

void init_state(
    struct State* self,
    uint32_t len,
    float min_radius,
    float max_radius,
    float min_px,
    float max_px,
    float min_py,
    float max_py,
    float min_vx,
    float max_vx,
    float min_vy,
    float max_vy
) {
    _mm_setcsr(_mm_getcsr() | _MM_MASK_DIV_ZERO);
    self->len = len;
    self->r  = aligned_alloc(64, len * sizeof *self->r);
    self->px = aligned_alloc(64, len * sizeof *self->px);
    self->py = aligned_alloc(64, len * sizeof *self->py);
    self->vx = aligned_alloc(64, len * sizeof *self->vx);
    self->vy = aligned_alloc(64, len * sizeof *self->vy);

    for (uint32_t i = 0; i < len; i++) {
        self->r[i] = rand_float(min_radius, max_radius);
        self->px[i] = rand_float(min_px, max_px);
        self->py[i] = rand_float(min_py, max_py);
        self->vx[i] = rand_float(min_vx, max_vx);
        self->vy[i] = rand_float(min_vy, max_vy);
    }
}

void destroy_state(struct State* self) {
    free(self->r);
    free(self->px);
    free(self->py);
    free(self->vx);
    free(self->vy);
}

static inline __v8su expand_mask8(uint8_t mask) {
    __m256 const lookup = _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
    __m256 const big_mask = _mm256_set1_epi32(mask);
    return _mm256_cmpeq_epi32(lookup, _mm256_and_si256(lookup, big_mask));
}

static inline void compress(uint32_t* target, __v8su mask, __v8su values) {
    uint32_t count = 0;

    for (size_t i = 0; i < 8; i++) {
        uint32_t const mask_value = ((uint32_t*) &mask)[i];
        uint32_t const value = ((uint32_t*) &values)[i];

        if (mask_value != 0) {
            target[count++] = value;
        }
    }
}

struct IndexPairVec* find_collisions(struct State const* self) {
    static struct IndexPairVec pairs = {0};
    __v8su const offsets = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    uint32_t i;
    for (i = 0; i + 7 < self->len; i += 8) {
        __v8su const is = i + offsets;

        __v8sf const pxs = _mm256_load_ps(&self->px[i]);
        __v8sf const pys = _mm256_load_ps(&self->py[i]);
        __v8sf const rs  = _mm256_load_ps(&self->r[i]);

        for (uint32_t j = i; j < self->len; j++) {
            uint8_t bit_mask = 0xFF;

            if (j == i + 8 - 1) {
                continue;
            }

            if (j >= i && j < i + 8) {
                bit_mask &= (0xFF << (j - i + 1));
            }

            __v8su mask = expand_mask8(bit_mask);

            float const px = self->px[j];
            float const py = self->py[j];
            float const r = self->r[j];

            __v8sf const dx = pxs - px;
            __v8sf const dy = pys - py;
            __v8sf const distance_squared = dy*dy + dx*dx;
            __v8sf const radius_sum = rs + r;
            __v8sf const distance_squared_for_collision = radius_sum * radius_sum;

            mask &= (__v8su) _mm256_cmp_ps(distance_squared, distance_squared_for_collision, _CMP_LE_OQ);

            if (_mm256_testz_si256(mask, mask)) {
                continue;
            }

            __v8su const js = _mm256_set1_epi32(j);

            bit_mask = _mm256_movemask_ps(mask);
            uint32_t const count = _mm_popcnt_u32(bit_mask);
            IndexPairVec_reserve_additional(&pairs, count);

            compress(pairs.is + pairs.count, mask, is);
            compress(pairs.js + pairs.count, mask, js);
            pairs.count += count;
        }
    }

    for (; i < self->len; i++) {
        float const pxs = self->px[i];
        float const pys = self->py[i];
        float const rs  = self->r[i];

        for (uint32_t j = i; j < self->len; j++) {
            float const px = self->px[j];
            float const py = self->py[j];
            float const r = self->r[j];

            float const dx = pxs - px;
            float const dy = pys - py;
            float const distance_squared = dy*dy + dx*dx;
            float const radius_sum = rs + r;
            float const distance_squared_for_collision = radius_sum * radius_sum;

            if (distance_squared > distance_squared_for_collision) {
                continue;
            }

            IndexPairVec_reserve_additional(&pairs, 1);

            pairs.is[pairs.count] = i;
            pairs.js[pairs.count] = j;
            pairs.count++;
        }
    }

    return &pairs;
}

void update_positions(struct State const* self, float dt) {
    __m256 const dts = _mm256_set1_ps(dt);

    uint32_t i;

    for (i = 0; i + 7 < self->len; i += 8) {
        __v8sf px = _mm256_load_ps(&self->px[i]);
        __v8sf py = _mm256_load_ps(&self->py[i]);
        __v8sf const vx = _mm256_load_ps(&self->vx[i]);
        __v8sf const vy = _mm256_load_ps(&self->vy[i]);

        px = _mm256_fmadd_ps(vx, dts, px);
        py = _mm256_fmadd_ps(vy, dts, py);

        _mm256_storeu_ps(&self->px[i], px);
        _mm256_storeu_ps(&self->py[i], py);
    }

    for (; i < self->len; i++) {
        float const vx = self->vx[i];
        float const vy = self->vy[i];
        self->px[i] += vx * dt;
        self->py[i] += vy * dt;
    }
}

void update_wall_collisions(struct State const* self) {
    float const single_max_x = GetScreenWidth();
    float const single_max_y = GetScreenHeight();

    __v8sf const max_x = _mm256_set1_ps(single_max_x);
    __v8sf const max_y = _mm256_set1_ps(single_max_y);

    uint32_t i;
    for (i = 0; i + 7 < self->len; i += 8) {
        __v8sf const r  = _mm256_load_ps(&self->r[i]);
        __v8sf const px = _mm256_load_ps(&self->px[i]);
        __v8sf const py = _mm256_load_ps(&self->py[i]);
        __v8sf const vx = _mm256_load_ps(&self->vx[i]);
        __v8sf const vy = _mm256_load_ps(&self->vy[i]);

        __m256 cmp_mask;
        __v8sf abs;

        cmp_mask = _mm256_cmp_ps(px, r, _CMP_LT_OQ);
        _mm256_maskstore_ps(&self->px[i], cmp_mask, r);
        abs = _mm256_and_ps(vx, _mm256_set1_epi32(0x7FFFFFFF));
        _mm256_maskstore_ps(&self->vx[i], cmp_mask, abs);

        cmp_mask = _mm256_cmp_ps(px + r, max_x, _CMP_GT_OQ);
        _mm256_maskstore_ps(&self->px[i], cmp_mask, max_x - r);
        abs = _mm256_and_ps(vx, _mm256_set1_epi32(0x80000000));
        _mm256_maskstore_ps(&self->vx[i], cmp_mask, abs);

        cmp_mask = _mm256_cmp_ps(py, r, _CMP_LT_OQ);
        _mm256_maskstore_ps(&self->py[i], cmp_mask, r);
        abs = _mm256_and_ps(vy, _mm256_set1_epi32(0x7FFFFFFF));
        _mm256_maskstore_ps(&self->vy[i], cmp_mask, abs);

        cmp_mask = _mm256_cmp_ps(py + r, max_y, _CMP_GT_OQ);
        _mm256_maskstore_ps(&self->py[i], cmp_mask, max_y - r);
        abs = _mm256_and_ps(vy, _mm256_set1_epi32(0x80000000));
        _mm256_maskstore_ps(&self->vy[i], cmp_mask, abs);
    }

    for (; i < self->len; i++) {
        float const r  = self->r[i];
        float const px = self->px[i];
        float const py = self->py[i];
        float const vx = self->vx[i];
        float const vy = self->vy[i];

        if (px < r) {
            self->px[i] = r;
            self->vx[i] = fabsf(vx);
        }

        if (px + r > single_max_x) {
            self->px[i] = single_max_x - r;
            self->vx[i] = -fabsf(vx);
        }

        if (py < r) {
            self->py[i] = r;
            self->vy[i] = fabsf(vy);
        }

        if (py + r > single_max_y) {
            self->py[i] = single_max_y - r;
            self->vy[i] = -fabsf(vy);
        }
    }
}

static inline void scatter(float* addr, uint32_t indices[], float values[], uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        addr[indices[i]] = values[i];
    }
}

void update_static_collisions(
    struct State const* self,
    struct IndexPairVec const* collisions
) {
    uint32_t index;

    for (index = 0; index + 7 < collisions->count; index += 8) {
        __v8su const is = _mm256_load_si256((void*) &collisions->is[index]);
        __v8su const js = _mm256_load_si256((void*) &collisions->js[index]);

        __v8sf px1 = _mm256_i32gather_ps(self->px, is, sizeof *self->px);
        __v8sf px2 = _mm256_i32gather_ps(self->px, js, sizeof *self->px);
        __v8sf py1 = _mm256_i32gather_ps(self->py, is, sizeof *self->py);
        __v8sf py2 = _mm256_i32gather_ps(self->py, js, sizeof *self->py);

        __v8sf const r1 = _mm256_i32gather_ps(self->r, is, sizeof *self->r);
        __v8sf const r2 = _mm256_i32gather_ps(self->r, js, sizeof *self->r);

        __v8sf const dx = px1 - px2;
        __v8sf const dy = py1 - py2;

        __v8sf const rdistance = _mm256_rsqrt_ps(dx*dx + dy*dy);
        __v8sf const radius_sum = r1 + r2;

        __v8sf const not_really_overlap = 0.5 * (1. - radius_sum * rdistance);
        __v8sf const offset_x = not_really_overlap * dx;
        __v8sf const offset_y = not_really_overlap * dy;

        px1 -= offset_x;
        py1 -= offset_y;
        scatter(self->px, (uint32_t*) &is, (float*) &px1, 8);
        scatter(self->py, (uint32_t*) &is, (float*) &py1, 8);

        px2 += offset_x;
        py2 += offset_y;
        scatter(self->px, (uint32_t*) &js, (float*) &px2, 8);
        scatter(self->py, (uint32_t*) &js, (float*) &py2, 8);
    }

    for (; index < collisions->count; index++) {
        uint32_t const i = collisions->is[index];
        uint32_t const j = collisions->js[index];

        float px1 = self->px[i];
        float px2 = self->px[j];
        float py1 = self->py[i];
        float py2 = self->py[j];

        float const r1 = self->r[i];
        float const r2 = self->r[j];

        float const dx = px1 - px2;
        float const dy = py1 - py2;

        float const rdistance = sqrtf(dx*dx + dy*dy);
        float const radius_sum = r1 + r2;

        float const not_really_overlap = 0.5 * (1. - radius_sum * rdistance);
        float const offset_x = not_really_overlap * dx;
        float const offset_y = not_really_overlap * dy;

        px1 -= offset_x;
        py1 -= offset_y;
        self->px[i] = px1;
        self->py[i] = py1;

        px2 += offset_x;
        py2 += offset_y;
        self->px[j] = px2;
        self->py[j] = py2;
    }
}

void update_dynamic_collisions(
    struct State const* self,
    struct IndexPairVec const* collisions
) {
    uint32_t index;
    for (index = 0; index + 7 < collisions->count; index += 8) {
        __v8su const is = _mm256_load_si256((void*) &collisions->is[index]);
        __v8su const js = _mm256_load_si256((void*) &collisions->js[index]);

        __v8sf const px1 = _mm256_i32gather_ps(self->px, is, sizeof *self->px);
        __v8sf const px2 = _mm256_i32gather_ps(self->px, js, sizeof *self->px);
        __v8sf const py1 = _mm256_i32gather_ps(self->py, is, sizeof *self->py);
        __v8sf const py2 = _mm256_i32gather_ps(self->py, js, sizeof *self->py);

        __v8sf vx1 = _mm256_i32gather_ps(self->vx, is, sizeof *self->vx);
        __v8sf vx2 = _mm256_i32gather_ps(self->vx, js, sizeof *self->vx);
        __v8sf vy1 = _mm256_i32gather_ps(self->vy, is, sizeof *self->vy);
        __v8sf vy2 = _mm256_i32gather_ps(self->vy, js, sizeof *self->vy);

        __v8sf const r1 = _mm256_i32gather_ps(self->r, is, sizeof *self->r);
        __v8sf const r2 = _mm256_i32gather_ps(self->r, js, sizeof *self->r);

        __v8sf const dx = px2 - px1;
        __v8sf const dy = py2 - py1;

        __v8sf const inv_distance = _mm256_rsqrt_ps(dx*dx + dy*dy);

        __v8sf const nx = inv_distance * dx;
        __v8sf const ny = inv_distance * dy;

        __v8sf const kx = vx1 - vx2;
        __v8sf const ky = vy1 - vy2;
        __v8sf const p  = 2.0 * (nx*kx + ny*ky) / (r1 + r2);

        vx1 -= p * r2 * nx;
        vy1 -= p * r2 * ny;
        vx2 += p * r1 * nx;
        vy2 += p * r1 * ny;

        scatter(self->vx, (uint32_t*) &is, (float*) &vx1, sizeof *self->vx);
        scatter(self->vy, (uint32_t*) &is, (float*) &vy1, sizeof *self->vy);
        scatter(self->vx, (uint32_t*) &js, (float*) &vx2, sizeof *self->vx);
        scatter(self->vy, (uint32_t*) &js, (float*) &vy2, sizeof *self->vy);
    }

    for (; index < collisions->count; index++) {
        uint32_t const i = collisions->is[index];
        uint32_t const j = collisions->js[index];

        float const px1 = self->px[i];
        float const py1 = self->py[i];
        float const px2 = self->px[j];
        float const py2 = self->py[j];

        float vx1 = self->vx[i];
        float vy1 = self->vy[i];
        float vx2 = self->vx[j];
        float vy2 = self->vy[j];

        float const r1 = self->r[i];
        float const r2 = self->r[j];

        float const dx = px2 - px1;
        float const dy = py2 - py1;

        float const inv_distance = 1. / sqrtf(dx*dx + dy*dy);

        float const nx = inv_distance * dx;
        float const ny = inv_distance * dy;

        float const kx = vx1 - vx2;
        float const ky = vy1 - vy2;
        float const p  = 2.0 * (nx*kx + ny*ky) / (r1 + r2);

        vx1 -= p * r2 * nx;
        vy1 -= p * r2 * ny;
        vx2 += p * r1 * nx;
        vy2 += p * r1 * ny;

        self->vx[i] = vx1;
        self->vy[i] = vy1;
        self->vx[j] = vx2;
        self->vy[j] = vy2;
    }
}

void update(struct State const* self, float dt) {
    update_positions(self, dt);
    update_wall_collisions(self);

    struct IndexPairVec* collisions = find_collisions(self);
    update_static_collisions(self, collisions);
    update_dynamic_collisions(self, collisions);

    collisions->count = 0;
}

void draw(struct State const* self) {
    for (uint32_t i = 0; i < self->len; i++) {
        float const r = self->r[i];
        float const px = self->px[i];
        float const py = self->py[i];
        DrawCircle(px, py, r, RED);
    }
}
