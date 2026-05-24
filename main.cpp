#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <omp.h> // Librería de OpenMP añadida

const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 500;

struct Pixel {
    unsigned char r, g, b;
};

void generarMandelbrot(std::vector<std::vector<Pixel>>& imagen) {
    // Directiva básica de OpenMP para paralelizar el bucle exterior
    #pragma omp parallel for
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double cr = -2.0 + (x * 3.0 / WIDTH);
            double ci = -1.2 + (y * 2.4 / HEIGHT);

            double zr = 0.0, zi = 0.0;
            int iter = 0;

            while (zr * zr + zi * zi <= 4.0 && iter < MAX_ITER) {
                double temp = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = temp;
                iter++;
            }

            if (iter == MAX_ITER) {
                imagen[y][x] = {0, 0, 0};
            } else {
                unsigned char c = static_cast<unsigned char>((iter * 255) / MAX_ITER);
                imagen[y][x] = {static_cast<unsigned char>(c * 3), static_cast<unsigned char>(c * 5), static_cast<unsigned char>(c * 7)};
            }
        }
    }
}

void aplicarFiltroConvolucion(const std::vector<std::vector<Pixel>>& origen, std::vector<std::vector<Pixel>>& destino) {
    const int K_SIZE = 5;
    float kernel[K_SIZE][K_SIZE] = {
        {1/273.0f, 4/273.0f,  7/273.0f,  4/273.0f,  1/273.0f},
        {4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f},
        {7/273.0f, 26/273.0f, 41/273.0f, 26/273.0f, 7/273.0f},
        {4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f},
        {1/273.0f, 4/273.0f,  7/273.0f,  4/273.0f,  1/273.0f}
    };
    int offset = K_SIZE / 2;

    // Directiva básica de OpenMP para paralelizar el filtro
    #pragma omp parallel for
    for (int y = offset; y < HEIGHT - offset; ++y) {
        for (int x = offset; x < WIDTH - offset; ++x) {
            float red = 0.0f, green = 0.0f, blue = 0.0f;

            for (int ky = 0; ky < K_SIZE; ++ky) {
                for (int kx = 0; kx < K_SIZE; ++kx) {
                    int px = x + kx - offset;
                    int py = y + ky - offset;

                    red += origen[py][px].r * kernel[ky][kx];
                    green += origen[py][px].g * kernel[ky][kx];
                    blue += origen[py][px].b * kernel[ky][kx];
                }
            }

            destino[y][x].r = static_cast<unsigned char>(std::min(std::max(red, 0.0f), 255.0f));
            destino[y][x].g = static_cast<unsigned char>(std::min(std::max(green, 0.0f), 255.0f));
            destino[y][x].b = static_cast<unsigned char>(std::min(std::max(blue, 0.0f), 255.0f));
        }
    }
}

void guardarImagenPPM(const std::string& nombreArchivo, const std::vector<std::vector<Pixel>>& imagen) {
    std::ofstream archivo(nombreArchivo, std::ios::binary);
    archivo << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int y = 0; y < HEIGHT; ++y) {
        archivo.write(reinterpret_cast<const char*>(imagen[y].data()), WIDTH * sizeof(Pixel));
    }
}

int main() {
    auto imagenBase = std::vector<std::vector<Pixel>>(HEIGHT, std::vector<Pixel>(WIDTH));
    auto imagenFiltrada = std::vector<std::vector<Pixel>>(HEIGHT, std::vector<Pixel>(WIDTH));

    std::cout << "Iniciando procesamiento paralelo con OpenMP (Línea Base)..." << std::endl;

    auto inicio = std::chrono::high_resolution_clock::now();

    generarMandelbrot(imagenBase);
    aplicarFiltroConvolucion(imagenBase, imagenFiltrada);

    auto fin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempo = fin - inicio;

    std::cout << "Tiempo de ejecucion paralelo: " << tiempo.count() << " segundos." << std::endl;

    guardarImagenPPM("mandelbrot_8k_filtrado_omp.ppm", imagenFiltrada);
    std::cout << "Imagen guardada exitosamente." << std::endl;

    return 0;
}
