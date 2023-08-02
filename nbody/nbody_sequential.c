#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define N 10000
#define G 6.67e-11
#define TIMESTEP 0.25
#define NSTEPS 5

/* 
 * body data structure
 */
struct body_s {
    double x;
    double y;
    double z;
    double dx;
    double dy;
    double dz;
    double mass;
};
typedef struct body_s body_t;


/* 
 * function prototypes
 */
void init(void);
double dist(double dx, double dy, double dz);


body_t bodies[N];  // array of N-bodies at timestep t
body_t next[N];    // array of N-bodies at timestep t+1



/**
 * init - give the planets initial values for position, velocity, mass
 */
void init(void) {
    for (int i=0; i<N; i++) {
        bodies[i].x = 100.0 * (i + 0.1);
        bodies[i].y = 200.0 * (i + 0.1);
        bodies[i].z = 300.0 * (i + 0.1);
        bodies[i].dx = i + 400.0;
        bodies[i].dy = i + 500.0;
        bodies[i].dz = i + 600.0;
        bodies[i].mass = 10e14 * (i + 100.2);
    }
}



/**
 * dist - determine the distance between two bodies
 *    @param dx - distance in the x dimension
 *    @param dy - distance in the y dimension
 *    @param dz - distance in the z dimension
 *    @return distance 
 */
double dist(double dx, double dy, double dz) {
    return sqrt((dx*dx) + (dy*dy) + (dz*dz));;
}




/**
 * computeforce - compute the superposed forces on one body
 *   @param me     - the body to compute forces on at time t
 *   @param nextme - the body at time t+1
 */
void next_state(body_t *me, body_t *nextme) {
    // double d, f;        // distance, force
    // double dx, dy, dz;  // position deltas
    // double fx, fy, fz;  // force components
    // double ax, ay, az;  // acceleration components

    double d_reciprocal;    // the inverse of distance (1 / d)
    double a_over_d;        // acceleration divided by distance (a / d)
    double dx, dy, dz;      // position deltas
    double ax, ay, az;      // acceleration components

    ax = ay = az = 0.0;

    // for every other body relative to me
    for (int i=0; i<N; i++) {

        // compute the distances in each dimension
        dx = me->x - bodies[i].x;
        dy = me->y - bodies[i].y;
        dz = me->z - bodies[i].z;

        // compute the distance magnitude
        d_reciprocal = 1.0 / sqrt((dx*dx) + (dy*dy) + (dz*dz));

        // skip over ourselves (d==0)
        if ((dx != 0 || dy != 0 || dz != 0)) { 

            // F = G m1 m2 / r^2
            a_over_d = G  * bodies[i].mass * d_reciprocal * d_reciprocal * d_reciprocal;

            // compute force components in each dimension
            ax += a_over_d * dx;  
            ay += a_over_d * dy;
            az += a_over_d * dz;
        }
    }
    // update the body velocity at time t+1
    nextme->dx = me->dx + (TIMESTEP * ax);
    nextme->dy = me->dy + (TIMESTEP * ay);
    nextme->dz = me->dz + (TIMESTEP * az);

    // update the body position at t+1
    nextme->x = me->x + (TIMESTEP * me->dx);
    nextme->y = me->y + (TIMESTEP * me->dy);
    nextme->z = me->z + (TIMESTEP * me->dz);

    // copy over the mass
    nextme->mass = me-> mass;
}



/**
 *  print_body - prints a body for debugging
 *    @param b - body to print
 */
void print_body(body_t *b) {
    printf("x: %7.3f y: %7.3f z: %7.3f dx: %7.3f dy: %7.3f dz: %7.3f mass: %7.3f\n",
            b->x, b->y, b->z, b->dx, b->dy, b->dz, b->mass);
}

/**
 * main
 */
int main(int argc, char **argv) {
    clock_t start, tsstart;

    // setup initial conditions
    init();

    for (int i = 0; i < 10; i++) {
        print_body(&bodies[i]);
    }

    printf("beginning N-body simulation of %d bodies.\n", N);

    start = clock();
    // for each timestep in the simulation
    for (int ts=0; ts<NSTEPS; ts++) {
        tsstart = clock();
        // for every body in the universe
        for (int i=0; i<N; i++) {
            next_state(&bodies[i], &next[i]);
        }

        // copy the t+1 state to be the new time t
        for (int i=0; i< N; i++) {
            memcpy(&bodies[i], &next[i], sizeof(body_t));
        }

        printf("timestep %d complete: %.7f s\n", ts, (double) (clock() - tsstart) / CLOCKS_PER_SEC);

        for (int i = 0; i < 10; i++) {
            print_body(&bodies[i]);
        }
    }
    printf("simulation complete: %.7f s\n", (double) (clock() - start) / CLOCKS_PER_SEC);
    return 0;
}