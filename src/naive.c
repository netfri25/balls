#include "raylib.h"

#include "common.h"

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

struct IndexPairVec {
    size_t count;
    size_t capacity;

    struct Pair {
        size_t i;
        size_t j;
    }* items;
};


struct Circle {
    float r;
    float px;
    float py;
    float vx;
    float vy;
};

struct State {
    struct Sampler sampler;
    size_t len;
    struct Circle* circles;
};

void init_state(
    struct State* self,
    size_t len,
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
    self->sampler = (struct Sampler){0};
    self->len = len;
    self->circles = malloc(len * sizeof *self->circles);

    for (size_t i = 0; i < len; i++) {
        self->circles[i] = (struct Circle) {
            .r = rand_float(min_radius, max_radius),
            .px = rand_float(min_px, max_px),
            .py = rand_float(min_py, max_py),
            .vx = rand_float(min_vx, max_vx),
            .vy = rand_float(min_vy, max_vy),
        };
    }
}

void destroy_state(struct State* self) {
    free(self->circles);
}

struct IndexPairVec find_collisions(struct State const* const self) {
    struct IndexPairVec pairs = {0};

    for (size_t i = 0; i < self->len; i++) {
        struct Circle const* const c1 = &self->circles[i];

        for (size_t j = i + 1; j < self->len; j++) {
            struct Circle const* const c2 = &self->circles[j];

            float const dx = c2->px - c1->px;
            float const dy = c2->py - c1->py;
            float const distance_squared = dx*dx + dy*dy;
            float const radius_sum = c2->r + c1->r;
            float const distance_squared_for_collision = radius_sum * radius_sum;

            if (distance_squared > distance_squared_for_collision)
                continue;

            struct Pair pair = {
                .i = i,
                .j = j,
            };

            da_append(&pairs, pair);
        }
    }

    return pairs;
}

void update_positions(struct State const* self, float dt) {
    for (size_t i = 0; i < self->len; i++) {
        struct Circle* const circle = &self->circles[i];
        circle->px += circle->vx * dt;
        circle->py += circle->vy * dt;
    }
}

void update_wall_collisions(struct State const* self) {
    float const max_x = GetScreenWidth();
    float const max_y = GetScreenHeight();

    for (size_t i = 0; i < self->len; i++) {
        struct Circle* const circle = &self->circles[i];

        if (circle->px < circle->r) {
            circle->px = circle->r;
            circle->vx = fabs(circle->vx);
        }

        if (circle->py < circle->r) {
            circle->py = circle->r;
            circle->vy = fabs(circle->vy);
        }

        if (circle->px + circle->r > max_x) {
            circle->px = max_x - circle->r;
            circle->vx = -fabs(circle->vx);
        }

        if (circle->py + circle->r > max_y) {
            circle->py = max_y - circle->r;
            circle->vy = -fabs(circle->vy);
        }
    }
}

void update_static_collisions(
    struct State const* self,
    struct IndexPairVec const* collisions
) {
    for (size_t i = 0; i < collisions->count; i++) {
        struct Pair const collision_pair = collisions->items[i];
        struct Circle* const c1 = &self->circles[collision_pair.i];
        struct Circle* const c2 = &self->circles[collision_pair.j];

        float const dx = c1->px - c2->px;
        float const dy = c1->py - c2->py;

        // TODO: maybe open the parens and optimize it to be rsqrt instead
        float const distance = sqrtf(dx*dx + dy*dy);
        float const overlap = (distance - c1->r - c2->r) / 2.;

        // NOTE: parens here
        float const offset_x = overlap * dx / distance;
        float const offset_y = overlap * dy / distance;

        c1->px -= offset_x;
        c1->py -= offset_y;

        c2->px += offset_x;
        c2->py += offset_y;
    }
}

void update_dynamic_collisions(
    struct State const* self,
    struct IndexPairVec const* collisions
) {
    for (size_t i = 0; i < collisions->count; i++) {
        struct Pair const collision_pair = collisions->items[i];
        struct Circle* const c1 = &self->circles[collision_pair.i];
        struct Circle* const c2 = &self->circles[collision_pair.j];

        float const dx = c1->px - c2->px;
        float const dy = c1->py - c2->py;

        float const distance = sqrtf(dx*dx + dy*dy);

        float const nx = (c2->px - c1->px) / distance;
        float const ny = (c2->py - c1->py) / distance;

        float const kx = (c1->vx - c2->vx);
        float const ky = (c1->vy - c2->vy);
        float const p = 2.0 * (nx * kx + ny * ky) / (c1->r + c2->r);
        c1->vx = c1->vx - p * c2->r * nx;
        c1->vy = c1->vy - p * c2->r * ny;
        c2->vx = c2->vx + p * c1->r * nx;
        c2->vy = c2->vy + p * c1->r * ny;
    }
}

void update(struct State const* self, float dt) {
    update_positions(self, dt);
    update_wall_collisions(self);

    struct IndexPairVec collisions = find_collisions(self);
    update_static_collisions(self, &collisions);
    update_dynamic_collisions(self, &collisions);

    da_free(&collisions);
}

void draw(struct State const* self) {
    for (size_t i = 0; i < self->len; i++) {
        struct Circle const* const circle = &self->circles[i];
        DrawCircle(
            round(circle->px),
            round(circle->py),
            circle->r,
            RED
        );
    }
}
