#include "raylib.h"

#include <stdint.h>
#include <stdio.h>
#include <assert.h>


#ifdef NAIVE
#include "naive.c"
#else
#include "vect.c"
#endif

#define ITERATIONS 20

int main(void) {
    SetTraceLogLevel(LOG_ERROR);

    // srand(time(NULL));
    InitWindow(800, 800, "window");
    SetTargetFPS(GetMonitorRefreshRate(0));
    SetWindowState(FLAG_WINDOW_RESIZABLE);

    // wait for the window to initialize
    BeginDrawing();
    EndDrawing();
    BeginDrawing();
    EndDrawing();

    struct State state;
    float const w = GetScreenWidth();
    float const h = GetScreenHeight();

    init_state(
        &state,
        CIRCLE_COUNT,

        // radius range
        MIN_RADIUS,
        MAX_RADIUS,

        // random px range
        MAX_RADIUS,
        w - MAX_RADIUS,

        // random py range
        MAX_RADIUS,
        h - MAX_RADIUS,

        // random vx range
        -100.,
        100.,

        // random vy range
        -100.,
        100.
    );

    while (!WindowShouldClose()) {
        float const total_dt = GetFrameTime();
        float const dt = total_dt / ITERATIONS;
        for (uint32_t i = 0; i < ITERATIONS; i++) {
            MEASURE(&state.sampler, update(&state, dt));
        }

        BeginDrawing();
            ClearBackground(BLACK);
            draw(&state);
            DrawFPS(0, 0);
        EndDrawing();
    }

    printf("update avg:     %.03fms\n", state.sampler.avg);
    printf("update lowest:  %.03fms\n", state.sampler.lowest);
    printf("update highest: %.03fms\n", state.sampler.highest);

    destroy_state(&state);
    CloseWindow();
}
