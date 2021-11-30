#ifndef KERNEL_H
#define KERNEL_H

struct uchar4;
struct int2;

#include <stdio.h>

#include <SceneHandler.h>
#include <Camera.h>

using namespace glm;

void drawScene(uchar4 *d_out, int w, int h, SceneHandler sc, Camera cam);

#endif
