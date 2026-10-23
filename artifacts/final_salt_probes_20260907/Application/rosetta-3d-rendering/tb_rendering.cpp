#include "ap_int.h"
#include <fstream>
#include <iostream>
#include <string>

const int MAX_X = 256;
const int MAX_Y = 256;
const int NUM_FB = MAX_X * MAX_Y / 4;
const int NUM_3D_TRI = 3192;

typedef ap_uint<8>  bit8;
typedef ap_uint<32> bit32;

typedef struct {
  bit8 x0; bit8 y0; bit8 z0;
  bit8 x1; bit8 y1; bit8 z1;
  bit8 x2; bit8 y2; bit8 z2;
} Triangle_3D;

/* Canonical Cornell Rosetta 3D-rendering "bear" dataset:
 * Triangle_3D triangle_3ds[3192].
 */
#include "input_data.h"

void rendering(bit32 input[3 * NUM_3D_TRI], bit32 output[NUM_FB]);

static int check_golden(const bit32 output[NUM_FB])
{
    std::ifstream golden("outputs_golden.txt");
    if (!golden.is_open()) {
        std::cerr << "ERROR: cannot open outputs_golden.txt\n";
        return 1;
    }

    std::string line;
    if (!std::getline(golden, line)) {
        std::cerr << "ERROR: empty outputs_golden.txt\n";
        return 1;
    }

    int errors = 0;

    /* Rosetta's check_result.cpp prints y from 255 down to 0 and
     * x from 0 up to 255. Each nonzero rendered pixel has color 100.
     */
    for (int row = 0; row < MAX_Y; ++row) {
        if (!std::getline(golden, line) || (int)line.size() < MAX_X) {
            std::cerr << "ERROR: malformed golden image at row " << row << "\n";
            return 1;
        }

        const int y = MAX_Y - 1 - row;

        for (int x = 0; x < MAX_X; ++x) {
            const int word_idx = x * (MAX_Y / 4) + y / 4;
            const int byte_idx = y & 3;
            bit32 word = output[word_idx];
            bit8 pixel = word.range(byte_idx * 8 + 7, byte_idx * 8);

            const int expected = (line[x] == '1') ? 100 : 0;
            const int got = pixel.to_uint();

            if (got != expected) {
                if (errors < 20) {
                    std::cerr << "Mismatch x=" << x
                              << " y=" << y
                              << " expected=" << expected
                              << " got=" << got << "\n";
                }
                ++errors;
            }
        }
    }

    if (errors != 0) {
        std::cerr << "ROSETTA 3D RENDERING FAILED: "
                  << errors << " pixel mismatches\n";
        return 1;
    }

    std::cout << "ROSETTA 3D RENDERING PASSED\n";
    return 0;
}

int main()
{
    static bit32 input[3 * NUM_3D_TRI];
    static bit32 output[NUM_FB];

    if (num_3d_triangles != NUM_3D_TRI) {
        std::cerr << "ERROR: input_data.h contains "
                  << num_3d_triangles
                  << " triangles, expected " << NUM_3D_TRI << "\n";
        return 1;
    }

    /* Pack the canonical Rosetta host input exactly as the upstream
     * 3d_rendering_host.cpp does.
     */
    for (int i = 0; i < NUM_3D_TRI; ++i) {
        input[3 * i]     = 0;
        input[3 * i + 1] = 0;
        input[3 * i + 2] = 0;

        input[3 * i](7, 0)       = triangle_3ds[i].x0;
        input[3 * i](15, 8)      = triangle_3ds[i].y0;
        input[3 * i](23, 16)     = triangle_3ds[i].z0;
        input[3 * i](31, 24)     = triangle_3ds[i].x1;

        input[3 * i + 1](7, 0)   = triangle_3ds[i].y1;
        input[3 * i + 1](15, 8)  = triangle_3ds[i].z1;
        input[3 * i + 1](23, 16) = triangle_3ds[i].x2;
        input[3 * i + 1](31, 24) = triangle_3ds[i].y2;

        input[3 * i + 2](7, 0)   = triangle_3ds[i].z2;
        input[3 * i + 2](31, 8)  = 0;
    }

    for (int i = 0; i < NUM_FB; ++i)
        output[i] = 0;

    rendering(input, output);

    return check_golden(output);
}

