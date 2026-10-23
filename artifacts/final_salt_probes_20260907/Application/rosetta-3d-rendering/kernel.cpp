/* MailoHLS ASPLOS external-evaluation source.
 * Benchmark: rosetta-3d-rendering
 * Source basis: sharc-lab/HLSFactory Rosetta 3d_rendering, derived from Cornell Rosetta.
 * Header typedefs are flattened so this file is self-contained except for Vitis HLS ap_int.h.
 * Existing optimization pragmas were removed; Lk sites follow the MailoHLS external-action mapping.
 */

#include "ap_int.h"

const int MAX_X = 256;
const int MAX_Y = 256;
const int NUM_FB = MAX_X * MAX_Y / 4;
const int NUM_3D_TRI = 3192;

typedef ap_uint<1> bit1;
typedef ap_uint<2> bit2;
typedef ap_uint<8> bit8;
typedef ap_uint<16> bit16;
typedef ap_uint<32> bit32;

typedef struct {
  bit8 x0; bit8 y0; bit8 z0;
  bit8 x1; bit8 y1; bit8 z1;
  bit8 x2; bit8 y2; bit8 z2;
} Triangle_3D;

typedef struct {
  bit8 x0; bit8 y0;
  bit8 x1; bit8 y1;
  bit8 x2; bit8 y2;
  bit8 z;
} Triangle_2D;

typedef struct {
  bit8 x; bit8 y; bit8 z; bit8 color;
} CandidatePixel;

typedef struct {
  bit8 x; bit8 y; bit8 color;
} Pixel;

int check_clockwise(Triangle_2D triangle_2d)
{
  int cw;
  cw = (triangle_2d.x2 - triangle_2d.x0) * (triangle_2d.y1 - triangle_2d.y0)
     - (triangle_2d.y2 - triangle_2d.y0) * (triangle_2d.x1 - triangle_2d.x0);
  return cw;
}

void clockwise_vertices(Triangle_2D *triangle_2d)
{
  bit8 tmp_x, tmp_y;
  tmp_x = triangle_2d->x0;
  tmp_y = triangle_2d->y0;
  triangle_2d->x0 = triangle_2d->x1;
  triangle_2d->y0 = triangle_2d->y1;
  triangle_2d->x1 = tmp_x;
  triangle_2d->y1 = tmp_y;
}

bit1 pixel_in_triangle(bit8 x, bit8 y, Triangle_2D triangle_2d)
{
  int pi0, pi1, pi2;
  pi0 = (x - triangle_2d.x0) * (triangle_2d.y1 - triangle_2d.y0)
      - (y - triangle_2d.y0) * (triangle_2d.x1 - triangle_2d.x0);
  pi1 = (x - triangle_2d.x1) * (triangle_2d.y2 - triangle_2d.y1)
      - (y - triangle_2d.y1) * (triangle_2d.x2 - triangle_2d.x1);
  pi2 = (x - triangle_2d.x2) * (triangle_2d.y0 - triangle_2d.y2)
      - (y - triangle_2d.y2) * (triangle_2d.x0 - triangle_2d.x2);
  return (pi0 >= 0 && pi1 >= 0 && pi2 >= 0);
}

bit8 find_min(bit8 in0, bit8 in1, bit8 in2)
{
  if (in0 < in1) {
    if (in0 < in2) return in0;
    else return in2;
  } else {
    if (in1 < in2) return in1;
    else return in2;
  }
}

bit8 find_max(bit8 in0, bit8 in1, bit8 in2)
{
  if (in0 > in1) {
    if (in0 > in2) return in0;
    else return in2;
  } else {
    if (in1 > in2) return in1;
    else return in2;
  }
}

void projection(Triangle_3D triangle_3d, Triangle_2D *triangle_2d, bit2 angle)
{
  if (angle == 0) {
    triangle_2d->x0 = triangle_3d.x0; triangle_2d->y0 = triangle_3d.y0;
    triangle_2d->x1 = triangle_3d.x1; triangle_2d->y1 = triangle_3d.y1;
    triangle_2d->x2 = triangle_3d.x2; triangle_2d->y2 = triangle_3d.y2;
    triangle_2d->z = triangle_3d.z0 / 3 + triangle_3d.z1 / 3 + triangle_3d.z2 / 3;
  } else if (angle == 1) {
    triangle_2d->x0 = triangle_3d.x0; triangle_2d->y0 = triangle_3d.z0;
    triangle_2d->x1 = triangle_3d.x1; triangle_2d->y1 = triangle_3d.z1;
    triangle_2d->x2 = triangle_3d.x2; triangle_2d->y2 = triangle_3d.z2;
    triangle_2d->z = triangle_3d.y0 / 3 + triangle_3d.y1 / 3 + triangle_3d.y2 / 3;
  } else if (angle == 2) {
    triangle_2d->x0 = triangle_3d.z0; triangle_2d->y0 = triangle_3d.y0;
    triangle_2d->x1 = triangle_3d.z1; triangle_2d->y1 = triangle_3d.y1;
    triangle_2d->x2 = triangle_3d.z2; triangle_2d->y2 = triangle_3d.y2;
    triangle_2d->z = triangle_3d.x0 / 3 + triangle_3d.x1 / 3 + triangle_3d.x2 / 3;
  }
}

bit2 rasterization1(Triangle_2D triangle_2d, bit8 max_min[],
                    Triangle_2D *triangle_2d_same, bit16 max_index[])
{
  if (check_clockwise(triangle_2d) == 0) return 1;
  if (check_clockwise(triangle_2d) < 0) clockwise_vertices(&triangle_2d);

  triangle_2d_same->x0 = triangle_2d.x0; triangle_2d_same->y0 = triangle_2d.y0;
  triangle_2d_same->x1 = triangle_2d.x1; triangle_2d_same->y1 = triangle_2d.y1;
  triangle_2d_same->x2 = triangle_2d.x2; triangle_2d_same->y2 = triangle_2d.y2;
  triangle_2d_same->z = triangle_2d.z;

  max_min[0] = find_min(triangle_2d.x0, triangle_2d.x1, triangle_2d.x2);
  max_min[1] = find_max(triangle_2d.x0, triangle_2d.x1, triangle_2d.x2);
  max_min[2] = find_min(triangle_2d.y0, triangle_2d.y1, triangle_2d.y2);
  max_min[3] = find_max(triangle_2d.y0, triangle_2d.y1, triangle_2d.y2);
  max_min[4] = max_min[1] - max_min[0];
  max_index[0] = (max_min[1] - max_min[0]) * (max_min[3] - max_min[2]);
  return 0;
}

bit16 rasterization2(bit2 flag, bit8 max_min[], bit16 max_index[],
                     Triangle_2D triangle_2d_same, CandidatePixel fragment2[])
{
  if (flag) return 0;
  bit8 color = 100;
  bit16 i = 0;

  RAST2: for (bit16 k = 0; k < max_index[0]; k++) {
    bit8 x = max_min[0] + k % max_min[4];
    bit8 y = max_min[2] + k / max_min[4];
    if (pixel_in_triangle(x, y, triangle_2d_same)) {
      fragment2[i].x = x;
      fragment2[i].y = y;
      fragment2[i].z = triangle_2d_same.z;
      fragment2[i].color = color;
      i++;
    }
  }
  return i;
}

bit16 zculling(bit16 counter, CandidatePixel fragments[], bit16 size, Pixel pixels[])
{
  /*L2:*/ static bit8 z_buffer[MAX_X][MAX_Y];

  if (counter == 0) {
    ZCULLING_INIT_ROW:
    /*L2:*/ for (bit16 i = 0; i < MAX_X; i++) {
      ZCULLING_INIT_COL:
      /*L3:*/ for (bit16 j = 0; j < MAX_Y; j++) {
        z_buffer[i][j] = 255;
      }
    }
  }

  bit16 pixel_cntr = 0;
  ZCULLING:
  /*L4:*/ for (bit16 n = 0; n < size; n++) {
    if (fragments[n].z < z_buffer[fragments[n].y][fragments[n].x]) {
      pixels[pixel_cntr].x = fragments[n].x;
      pixels[pixel_cntr].y = fragments[n].y;
      pixels[pixel_cntr].color = fragments[n].color;
      pixel_cntr++;
      z_buffer[fragments[n].y][fragments[n].x] = fragments[n].z;
    }
  }
  return pixel_cntr;
}

void coloringFB(bit16 counter, bit16 size_pixels, Pixel pixels[],
                bit8 frame_buffer[MAX_X][MAX_Y])
{
  if (counter == 0) {
    COLORING_FB_INIT_ROW:
    /*L5:*/ for (bit16 i = 0; i < MAX_X; i++) {
      COLORING_FB_INIT_COL:
      /*L6:*/ for (bit16 j = 0; j < MAX_Y; j++) {
        frame_buffer[i][j] = 0;
      }
    }
  }

  COLORING_FB:
  /*L7:*/ for (bit16 i = 0; i < size_pixels; i++) {
    frame_buffer[pixels[i].x][pixels[i].y] = pixels[i].color;
  }
}

void output_FB(bit8 frame_buffer[MAX_X][MAX_Y], bit32 output[NUM_FB])
{
  bit32 out_FB = 0;
  OUTPUT_FB_ROW:
  /*L8:*/ for (bit16 i = 0; i < MAX_X; i++) {
    OUTPUT_FB_COL:
    /*L9:*/ for (bit16 j = 0; j < MAX_Y; j = j + 4) {
      out_FB(7, 0) = frame_buffer[i][j + 0];
      out_FB(15, 8) = frame_buffer[i][j + 1];
      out_FB(23, 16) = frame_buffer[i][j + 2];
      out_FB(31, 24) = frame_buffer[i][j + 3];
      output[i * MAX_Y / 4 + j / 4] = out_FB;
    }
  }
}

void rendering(bit32 input[3 * NUM_3D_TRI], bit32 output[NUM_FB])
{
  Triangle_3D triangle_3ds;
  Triangle_2D triangle_2ds;
  Triangle_2D triangle_2ds_same;

  bit16 size_fragment;
  /*L11:*/ CandidatePixel fragment[500];

  bit16 size_pixels;
  /*L12:*/ Pixel pixels[500];

  /*L13:*/ bit8 frame_buffer[MAX_X][MAX_Y];
  bit2 angle = 0;

  /*L14:*/ bit8 max_min[5];
  /*L15:*/ bit16 max_index[1];
  bit2 flag;

  TRIANGLES:
  /*L15:*/ for (bit16 i = 0; i < NUM_3D_TRI; i++) {
    bit32 input_lo = input[3 * i];
    bit32 input_mi = input[3 * i + 1];
    bit32 input_hi = input[3 * i + 2];

    triangle_3ds.x0 = input_lo(7, 0);
    triangle_3ds.y0 = input_lo(15, 8);
    triangle_3ds.z0 = input_lo(23, 16);
    triangle_3ds.x1 = input_lo(31, 24);
    triangle_3ds.y1 = input_mi(7, 0);
    triangle_3ds.z1 = input_mi(15, 8);
    triangle_3ds.x2 = input_mi(23, 16);
    triangle_3ds.y2 = input_mi(31, 24);
    triangle_3ds.z2 = input_hi(7, 0);

    projection(triangle_3ds, &triangle_2ds, angle);
    flag = rasterization1(triangle_2ds, max_min, &triangle_2ds_same, max_index);
    size_fragment = rasterization2(flag, max_min, max_index, triangle_2ds_same, fragment);
    size_pixels = zculling(i, fragment, size_fragment, pixels);
    coloringFB(i, size_pixels, pixels, frame_buffer);
  }

  output_FB(frame_buffer, output);
}
