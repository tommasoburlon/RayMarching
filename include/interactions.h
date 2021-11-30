#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#define WIDTH 1500
#define HEIGHT 1000
#define SPEED 10.0 // pixel increment for arrow keys
#define TITLE_STRING "Ray Marching"
#include <Camera.h>

int3 loc = {0, 0, 0};

bool dragMode = false; // mouse tracking mode
bool isPressed[256];
Camera cam;

void resetPointer(){
  if(abs(loc.x - WIDTH * 0.5) > WIDTH * 0.2 || abs(loc.y - HEIGHT * 0.5) > HEIGHT * 0.2){
    loc.x = WIDTH * 0.5;
    loc.y = HEIGHT * 0.5;
    loc.z = 2;
    glutWarpPointer(loc.x, loc.y);
  }
}

void initIO(){
  resetPointer();
  for(int i = 0; i < 256; i++)
    isPressed[i] = false;
}

void keydown(unsigned char key, int x, int y) {
  if (key == 27)  exit(0);

  isPressed[key] = true;
}

void keyup(unsigned char key, int x, int y) {
  isPressed[key] = false;
}

void mouseMove(int x, int y) {
   if (dragMode) return;


   float angY, angX;

   angX = cam.getAngX() - ((float)(y - loc.y)) / 100.0;
   angY = cam.getAngY() + ((float)(x - loc.x)) / 100.0;

   if(loc.z){
      loc.z--;
     return;
   }

   if (angX > 0.5 * M_PI) {
     angX = 0.5 * M_PI - 0.0001f;
   }
   else if (angX < -0.5 * M_PI) {
     angX = -0.5 * M_PI + 0.0001f;
   }

   cam.setAngX(angX);
   cam.setAngY(angY);

   loc.x = x;
   loc.y = y;

   resetPointer();
}

void mouseDrag(int x, int y) {
   if (!dragMode) return;

   loc.x = x;
   loc.y = y;
}

void handleSpecialKeypress(int key, int x, int y) {
}

#endif
