#include "raylib.h"

#include "common.h"

#include <assert.h>
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
        16 * sizeof *self->is
    );

    self->js = aligned_realloc(
        self->js,
        self->count    * sizeof *self->js,
        self->capacity * sizeof *self->js,
        16 * sizeof *self->js
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


struct IndexPairVec* find_collisions(struct State const* self) {
    static struct IndexPairVec pairs = {0};
    __v16su const offsets = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    for (uint32_t i = 0; i < self->len; i += 16) {
        uint32_t const leftover = self->len - i;
        uint32_t const bits_to_set = leftover >= 16 ? 16 : leftover;
        __mmask16 const valid_is = _mm512_int2mask((1 << bits_to_set) - 1);
        __v16su const is = i + offsets;

        __v16sf const pxs = _mm512_maskz_load_ps(valid_is, &self->px[i]);
        __v16sf const pys = _mm512_maskz_load_ps(valid_is, &self->py[i]);
        __v16sf const rs  = _mm512_maskz_load_ps(valid_is, &self->r[i]);

        for (uint32_t j = i; j < self->len; j++) {
            __mmask16 mask = valid_is;

            if (j == i + 15) {
                continue;
            }

            if (j >= i && j < i + 16) {
                mask &= _mm512_int2mask(0xFFFF << (j - i + 1));
            }

            float const px = self->px[j];
            float const py = self->py[j];
            float const r = self->r[j];

            __v16sf const dx = pxs - px;
            __v16sf const dy = pys - py;
            __v16sf const distance_squared = dy*dy + dx*dx;
            __v16sf const radius_sum = rs + r;
            __v16sf const distance_squared_for_collision = radius_sum * radius_sum;

            mask &= _mm512_cmp_ps_mask(distance_squared, distance_squared_for_collision, _CMP_LE_OQ);

            if (mask == 0) {
                continue;
            }

            __v16su const js = _mm512_set1_epi32(j);

            uint32_t const count = _mm_popcnt_u32(mask);
            IndexPairVec_reserve_additional(&pairs, count);

            _mm512_mask_compressstoreu_epi32(pairs.is + pairs.count, mask, is);
            _mm512_mask_compressstoreu_epi32(pairs.js + pairs.count, mask, js);
            pairs.count += count;
        }
    }

    return &pairs;
}

void update_positions(struct State const* self, float dt) {
    __m512 const dts = _mm512_set1_ps(dt);

    for (uint32_t i = 0; i < self->len; i += 16) {
        uint32_t const leftover = self->len - i;
        uint32_t const bits_to_set = leftover >= 16 ? 16 : leftover;
        __mmask16 const mask = _mm512_int2mask((1 << bits_to_set) - 1);

        __v16sf px = _mm512_maskz_load_ps(mask, &self->px[i]);
        __v16sf py = _mm512_maskz_load_ps(mask, &self->py[i]);
        __v16sf const vx = _mm512_maskz_load_ps(mask, &self->vx[i]);
        __v16sf const vy = _mm512_maskz_load_ps(mask, &self->vy[i]);

        px = _mm512_fmadd_ps(vx, dts, px);
        py = _mm512_fmadd_ps(vy, dts, py);

        _mm512_mask_storeu_ps(&self->px[i], mask, px);
        _mm512_mask_storeu_ps(&self->py[i], mask, py);
    }
}

void update_wall_collisions(struct State const* self) {
    __v16sf const max_x = _mm512_set1_ps(GetScreenWidth());
    __v16sf const max_y = _mm512_set1_ps(GetScreenHeight());

    for (uint32_t i = 0; i < self->len; i += 16) {
        uint32_t const leftover = self->len - i;
        uint32_t const bits_to_set = leftover >= 16 ? 16 : leftover;
        __mmask16 const mask = _mm512_int2mask((1 << bits_to_set) - 1);

        __v16sf r  = _mm512_maskz_load_ps(mask, &self->r[i]);
        __v16sf px = _mm512_maskz_load_ps(mask, &self->px[i]);
        __v16sf py = _mm512_maskz_load_ps(mask, &self->py[i]);
        __v16sf vx = _mm512_maskz_load_ps(mask, &self->vx[i]);
        __v16sf vy = _mm512_maskz_load_ps(mask, &self->vy[i]);

        __mmask16 cmp_mask;
        __v16sf abs;

        cmp_mask = mask & _mm512_cmp_ps_mask(px, r, _CMP_LT_OQ);
        _mm512_mask_storeu_ps(&self->px[i], cmp_mask, r);
        abs = _mm512_abs_ps(vx);
        _mm512_mask_storeu_ps(&self->vx[i], cmp_mask, abs);

        cmp_mask = mask & _mm512_cmp_ps_mask(py, r, _CMP_LT_OQ);
        if (cmp_mask) {
            _mm512_mask_storeu_ps(&self->py[i], cmp_mask, r);
            abs = _mm512_abs_ps(vy);
            _mm512_mask_storeu_ps(&self->vy[i], cmp_mask, abs);
        }

        cmp_mask = mask & _mm512_cmp_ps_mask(px + r, max_x, _CMP_GT_OQ);
        if (cmp_mask) {
            _mm512_mask_storeu_ps(&self->px[i], cmp_mask, max_x - r);
            abs = -_mm512_abs_ps(vx);
            _mm512_mask_storeu_ps(&self->vx[i], cmp_mask, abs);
        }

        cmp_mask = mask & _mm512_cmp_ps_mask(py + r, max_y, _CMP_GT_OQ);
        if (cmp_mask) {
            _mm512_mask_storeu_ps(&self->py[i], cmp_mask, max_y - r);
            abs = -_mm512_abs_ps(vy);
            _mm512_mask_storeu_ps(&self->vy[i], cmp_mask, abs);
        }
    }
}

void update_static_collisions(
    struct State const* self,
    struct IndexPairVec const* collisions
) {
    __v16sf const zerops = _mm512_setzero_ps();

    for (uint32_t index = 0; index < collisions->count; index += 16) {
        uint32_t const leftover = collisions->count - index;
        uint32_t const bits_to_set = leftover >= 16 ? 16 : leftover;
        __mmask16 const mask = _mm512_int2mask((1 << bits_to_set) - 1);

        __v16su const is = _mm512_maskz_load_epi32(mask, &collisions->is[index]);
        __v16su const js = _mm512_maskz_load_epi32(mask, &collisions->js[index]);

        __v16sf px1 = _mm512_mask_i32gather_ps(zerops, mask, is, self->px, sizeof *self->px);
        __v16sf px2 = _mm512_mask_i32gather_ps(zerops, mask, js, self->px, sizeof *self->px);
        __v16sf py1 = _mm512_mask_i32gather_ps(zerops, mask, is, self->py, sizeof *self->py);
        __v16sf py2 = _mm512_mask_i32gather_ps(zerops, mask, js, self->py, sizeof *self->py);

        __v16sf const r1 = _mm512_mask_i32gather_ps(zerops, mask, is, self->r, sizeof *self->r);
        __v16sf const r2 = _mm512_mask_i32gather_ps(zerops, mask, js, self->r, sizeof *self->r);

        __v16sf const dx = px1 - px2;
        __v16sf const dy = py1 - py2;

        __v16sf const rdistance = _mm512_rsqrt14_ps(dx*dx + dy*dy);
        __v16sf const radius_sum = r1 + r2;

        __v16sf const not_really_overlap = 0.5 * (1. - radius_sum * rdistance);
        __v16sf const offset_x = not_really_overlap * dx;
        __v16sf const offset_y = not_really_overlap * dy;

        px1 -= offset_x;
        py1 -= offset_y;
        _mm512_mask_i32scatter_ps(self->px, mask, is, px1, sizeof *self->px);
        _mm512_mask_i32scatter_ps(self->py, mask, is, py1, sizeof *self->py);

        px2 += offset_x;
        py2 += offset_y;
        _mm512_mask_i32scatter_ps(self->px, mask, js, px2, sizeof *self->px);
        _mm512_mask_i32scatter_ps(self->py, mask, js, py2, sizeof *self->py);
    }
}

void update_dynamic_collisions(
    struct State const* self,
    struct IndexPairVec const* collisions
) {
    __v16sf const zerops = _mm512_setzero_ps();

    for (uint32_t index = 0; index < collisions->count; index += 16) {
        uint32_t const leftover = collisions->count - index;
        uint32_t const bits_to_set = leftover >= 16 ? 16 : leftover;
        __mmask16 const mask = _mm512_int2mask((1 << bits_to_set) - 1);

        __v16su const is = _mm512_maskz_load_epi32(mask, &collisions->is[index]);
        __v16su const js = _mm512_maskz_load_epi32(mask, &collisions->js[index]);

        __v16sf const px1 = _mm512_mask_i32gather_ps(zerops, mask, is, self->px, sizeof *self->px);
        __v16sf const px2 = _mm512_mask_i32gather_ps(zerops, mask, js, self->px, sizeof *self->px);
        __v16sf const py1 = _mm512_mask_i32gather_ps(zerops, mask, is, self->py, sizeof *self->py);
        __v16sf const py2 = _mm512_mask_i32gather_ps(zerops, mask, js, self->py, sizeof *self->py);

        __v16sf vx1 = _mm512_mask_i32gather_ps(zerops, mask, is, self->vx, sizeof *self->vx);
        __v16sf vx2 = _mm512_mask_i32gather_ps(zerops, mask, js, self->vx, sizeof *self->vx);
        __v16sf vy1 = _mm512_mask_i32gather_ps(zerops, mask, is, self->vy, sizeof *self->vy);
        __v16sf vy2 = _mm512_mask_i32gather_ps(zerops, mask, js, self->vy, sizeof *self->vy);

        __v16sf const r1 = _mm512_mask_i32gather_ps(zerops, mask, is, self->r, sizeof *self->r);
        __v16sf const r2 = _mm512_mask_i32gather_ps(zerops, mask, js, self->r, sizeof *self->r);

        __v16sf const dx = px2 - px1;
        __v16sf const dy = py2 - py1;

        __v16sf const inv_distance = _mm512_rsqrt14_ps(dx*dx + dy*dy);

        __v16sf const nx = inv_distance * dx;
        __v16sf const ny = inv_distance * dy;

        __v16sf const kx = vx1 - vx2;
        __v16sf const ky = vy1 - vy2;
        __v16sf const p  = 2.0 * (nx*kx + ny*ky) / (r1 + r2);

        vx1 -= p * r2 * nx;
        vy1 -= p * r2 * ny;
        vx2 += p * r1 * nx;
        vy2 += p * r1 * ny;

        _mm512_mask_i32scatter_ps(self->vx, mask, is, vx1, sizeof *self->vx);
        _mm512_mask_i32scatter_ps(self->vy, mask, is, vy1, sizeof *self->vy);
        _mm512_mask_i32scatter_ps(self->vx, mask, js, vx2, sizeof *self->vx);
        _mm512_mask_i32scatter_ps(self->vy, mask, js, vy2, sizeof *self->vy);
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
