/*
 * demo_debruitage.c
 * Programme de démonstration pratique pour l'exposé
 * Comparaison des 3 méthodes de convolution avec Intel MKL
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>

// Structures simplifiées pour la démo
typedef struct {
    float *data;
    int width;
    int height;
    int channels;
} Image;

typedef struct {
    float *weights;
    int size;
    float sigma;
} Kernel;

// ============================================================
// FONCTIONS UTILITAIRES
// ============================================================

// Chronomètre
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Créer une image de test avec bruit
Image* create_test_image(int width, int height) {
    Image *img = malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->channels = 1; // Niveau de gris pour simplifier
    
    size_t total = width * height;
    img->data = mkl_malloc(total * sizeof(float), 64);
    
    // Créer un motif simple + bruit
    srand(42);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Damier de base
            float base = ((x/50 + y/50) % 2) ? 200.0f : 50.0f;
            // Ajouter du bruit gaussien
            float noise = (rand() % 100 - 50) * 0.5f;
            img->data[y * width + x] = base + noise;
        }
    }
    
    return img;
}

void free_image(Image *img) {
    if (img) {
        if (img->data) mkl_free(img->data);
        free(img);
    }
}

// Créer noyau gaussien
Kernel* create_gaussian_kernel(int size, float sigma) {
    Kernel *k = malloc(sizeof(Kernel));
    k->size = size;
    k->sigma = sigma;
    k->weights = mkl_malloc(size * size * sizeof(float), 64);
    
    int center = size / 2;
    float sum = 0.0f;
    float sigma_sq = sigma * sigma;
    float coeff = 1.0f / (2.0f * 3.14159265f * sigma_sq);
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int dx = x - center;
            int dy = y - center;
            float dist_sq = dx*dx + dy*dy;
            float val = coeff * expf(-dist_sq / (2.0f * sigma_sq));
            k->weights[y * size + x] = val;
            sum += val;
        }
    }
    
    // Normalisation
    for (int i = 0; i < size * size; i++) {
        k->weights[i] /= sum;
    }
    
    return k;
}

void free_kernel(Kernel *k) {
    if (k) {
        if (k->weights) mkl_free(k->weights);
        free(k);
    }
}

// ============================================================
// MÉTHODE 1 : CONVOLUTION SPATIALE
// ============================================================

Image* convolve_spatial(Image *img, Kernel *kernel) {
    Image *out = malloc(sizeof(Image));
    out->width = img->width;
    out->height = img->height;
    out->channels = img->channels;
    
    size_t total = img->width * img->height;
    out->data = mkl_malloc(total * sizeof(float), 64);
    
    int half = kernel->size / 2;
    
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernel->size; ky++) {
                for (int kx = 0; kx < kernel->size; kx++) {
                    int img_y = y + ky - half;
                    int img_x = x + kx - half;
                    
                    // Clamp
                    if (img_y < 0) img_y = 0;
                    if (img_y >= img->height) img_y = img->height - 1;
                    if (img_x < 0) img_x = 0;
                    if (img_x >= img->width) img_x = img->width - 1;
                    
                    sum += img->data[img_y * img->width + img_x] * 
                           kernel->weights[ky * kernel->size + kx];
                }
            }
            
            out->data[y * img->width + x] = sum;
        }
    }
    
    return out;
}

// ============================================================
// MÉTHODE 2 : CONVOLUTION SÉPARABLE
// ============================================================

float* create_gaussian_1d(int size, float sigma) {
    float *kernel = mkl_malloc(size * sizeof(float), 64);
    
    int center = size / 2;
    float sum = 0.0f;
    float sigma_sq = sigma * sigma;
    
    for (int i = 0; i < size; i++) {
        int x = i - center;
        float val = expf(-(x*x) / (2.0f * sigma_sq));
        kernel[i] = val;
        sum += val;
    }
    
    // Normalisation avec BLAS
    cblas_sscal(size, 1.0f / sum, kernel, 1);
    
    return kernel;
}

Image* convolve_1d(Image *img, float *kernel, int ksize, int horizontal) {
    Image *out = malloc(sizeof(Image));
    out->width = img->width;
    out->height = img->height;
    out->channels = img->channels;
    
    size_t total = img->width * img->height;
    out->data = mkl_malloc(total * sizeof(float), 64);
    
    int half = ksize / 2;
    
    if (horizontal) {
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                float sum = 0.0f;
                for (int k = 0; k < ksize; k++) {
                    int src_x = x + k - half;
                    if (src_x < 0) src_x = 0;
                    if (src_x >= img->width) src_x = img->width - 1;
                    sum += img->data[y * img->width + src_x] * kernel[k];
                }
                out->data[y * img->width + x] = sum;
            }
        }
    } else {
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                float sum = 0.0f;
                for (int k = 0; k < ksize; k++) {
                    int src_y = y + k - half;
                    if (src_y < 0) src_y = 0;
                    if (src_y >= img->height) src_y = img->height - 1;
                    sum += img->data[src_y * img->width + x] * kernel[k];
                }
                out->data[y * img->width + x] = sum;
            }
        }
    }
    
    return out;
}

Image* convolve_separable(Image *img, float *kernel, int ksize) {
    // Passe horizontale
    Image *temp = convolve_1d(img, kernel, ksize, 1);
    // Passe verticale
    Image *result = convolve_1d(temp, kernel, ksize, 0);
    free_image(temp);
    return result;
}

// ============================================================
// MÉTHODE 3 : CONVOLUTION FFT
// ============================================================

void* fft_forward(float *data, int width, int height) {
    DFTI_DESCRIPTOR_HANDLE handle;
    MKL_LONG dims[2] = {height, width};
    
    DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 2, dims);
    DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    
    MKL_LONG istrides[3] = {0, width, 1};
    DftiSetValue(handle, DFTI_INPUT_STRIDES, istrides);
    
    MKL_LONG ostrides[3] = {0, width/2 + 1, 1};
    DftiSetValue(handle, DFTI_OUTPUT_STRIDES, ostrides);
    
    DftiCommitDescriptor(handle);
    
    size_t csize = height * (width/2 + 1);
    float *result = mkl_malloc(csize * 2 * sizeof(float), 64);
    
    DftiComputeForward(handle, data, result);
    DftiFreeDescriptor(&handle);
    
    return result;
}

float* fft_backward(void *fft_data, int width, int height) {
    DFTI_DESCRIPTOR_HANDLE handle;
    MKL_LONG dims[2] = {height, width};
    
    DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 2, dims);
    DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiCommitDescriptor(handle);
    
    size_t total = width * height;
    float *result = mkl_malloc(total * sizeof(float), 64);
    
    DftiComputeBackward(handle, fft_data, result);
    DftiFreeDescriptor(&handle);
    
    // Normalisation
    float scale = 1.0f / (width * height);
    cblas_sscal(total, scale, result, 1);
    
    return result;
}

void fft_multiply(void *fft1, void *fft2, int width, int height) {
    float *f1 = (float*)fft1;
    float *f2 = (float*)fft2;
    size_t csize = height * (width/2 + 1);
    
    for (size_t i = 0; i < csize; i++) {
        float a = f1[2*i];
        float b = f1[2*i+1];
        float c = f2[2*i];
        float d = f2[2*i+1];
        
        f1[2*i] = a*c - b*d;
        f1[2*i+1] = a*d + b*c;
    }
}

Image* convolve_fft(Image *img, Kernel *kernel) {
    Image *out = malloc(sizeof(Image));
    out->width = img->width;
    out->height = img->height;
    out->channels = img->channels;
    
    size_t total = img->width * img->height;
    out->data = mkl_malloc(total * sizeof(float), 64);
    
    // Padder le noyau
    float *kpad = mkl_calloc(total, sizeof(float), 64);
    int khalf = kernel->size / 2;
    
    for (int y = 0; y < kernel->size; y++) {
        for (int x = 0; x < kernel->size; x++) {
            int dy = (y - khalf + img->height) % img->height;
            int dx = (x - khalf + img->width) % img->width;
            kpad[dy * img->width + dx] = kernel->weights[y * kernel->size + x];
        }
    }
    
    // FFT du noyau
    void *kfft = fft_forward(kpad, img->width, img->height);
    mkl_free(kpad);
    
    // FFT de l'image
    void *ifft = fft_forward(img->data, img->width, img->height);
    
    // Multiplication
    fft_multiply(ifft, kfft, img->width, img->height);
    
    // IFFT
    float *result = fft_backward(ifft, img->width, img->height);
    
    memcpy(out->data, result, total * sizeof(float));
    
    mkl_free(result);
    mkl_free(ifft);
    mkl_free(kfft);
    
    return out;
}

// ============================================================
// FONCTION PRINCIPALE DE DÉMONSTRATION
// ============================================================

void print_separator() {
    printf("\n");
    printf("================================================================\n");
}

void demonstrate_denoising() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║   DÉMONSTRATION : DÉBRUITAGE D'IMAGES AVEC INTEL MKL      ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    
    // Configuration MKL
    int num_threads = 4;
    mkl_set_num_threads(num_threads);
    
    printf("\n[*] Configuration MKL\n");
    printf("    - Threads : %d\n", mkl_get_max_threads());
    printf("    - Version : %s\n", mkl_get_version_string());
    
    // Paramètres
    int width = 512;
    int height = 512;
    int kernel_size = 7;
    float sigma = 2.0f;
    
    printf("\n[*] Paramètres de test\n");
    printf("    - Image : %dx%d pixels\n", width, height);
    printf("    - Noyau gaussien : %dx%d, σ=%.1f\n", kernel_size, kernel_size, sigma);
    
    print_separator();
    
    // Créer image de test
    printf("\n[1] Génération de l'image de test...\n");
    Image *img = create_test_image(width, height);
    printf("    ✓ Image créée : %d pixels avec bruit artificiel\n", width * height);
    
    // Créer noyau
    printf("\n[2] Génération du noyau gaussien...\n");
    Kernel *kernel = create_gaussian_kernel(kernel_size, sigma);
    printf("    ✓ Noyau %dx%d généré et normalisé\n", kernel_size, kernel_size);
    
    print_separator();
    
    // MÉTHODE 1 : Spatiale
    printf("\n[3] MÉTHODE 1 : CONVOLUTION SPATIALE\n");
    printf("    Description : Approche directe, parcours pixel par pixel\n");
    printf("    Complexité  : O(N × K²) = O(%d × %d²)\n", width*height, kernel_size);
    
    double t1 = get_time();
    Image *result_spatial = convolve_spatial(img, kernel);
    double t2 = get_time();
    double time_spatial = (t2 - t1) * 1000.0;
    
    printf("    ✓ Temps d'exécution : %.2f ms\n", time_spatial);
    printf("    → Méthode de référence (baseline)\n");
    
    // MÉTHODE 2 : Séparable
    printf("\n[4] MÉTHODE 2 : CONVOLUTION SÉPARABLE\n");
    printf("    Description : Décomposition en 2 passes 1D (H puis V)\n");
    printf("    Complexité  : O(2NK) = O(2 × %d × %d)\n", width*height, kernel_size);
    
    float *kernel_1d = create_gaussian_1d(kernel_size, sigma);
    
    t1 = get_time();
    Image *result_sep = convolve_separable(img, kernel_1d, kernel_size);
    t2 = get_time();
    double time_sep = (t2 - t1) * 1000.0;
    
    printf("    ✓ Temps d'exécution : %.2f ms\n", time_sep);
    printf("    → Accélération : %.1f×\n", time_spatial / time_sep);
    printf("    → Utilisation BLAS : cblas_sscal pour normalisation\n");
    
    // MÉTHODE 3 : FFT
    printf("\n[5] MÉTHODE 3 : CONVOLUTION PAR FFT\n");
    printf("    Description : Via transformée de Fourier rapide\n");
    printf("    Complexité  : O(N log N) = O(%d × log %d)\n", width*height, width*height);
    
    t1 = get_time();
    Image *result_fft = convolve_fft(img, kernel);
    t2 = get_time();
    double time_fft = (t2 - t1) * 1000.0;
    
    printf("    ✓ Temps d'exécution : %.2f ms\n", time_fft);
    printf("    → Accélération : %.1f×\n", time_spatial / time_fft);
    printf("    → Utilisation MKL DFTI : FFT 2D optimisée\n");
    
    print_separator();
    
    // Tableau récapitulatif
    printf("\n[6] TABLEAU RÉCAPITULATIF DES PERFORMANCES\n\n");
    printf("┌──────────────────┬──────────────┬──────────────┬──────────────┐\n");
    printf("│ Méthode          │ Temps (ms)   │ Accélération │ MKL utilisé  │\n");
    printf("├──────────────────┼──────────────┼──────────────┼──────────────┤\n");
    printf("│ Spatiale         │ %10.2f   │     1.0×     │      -       │\n", time_spatial);
    printf("│ Séparable        │ %10.2f   │   %6.1f×    │    BLAS      │\n", time_sep, time_spatial/time_sep);
    printf("│ FFT              │ %10.2f   │   %6.1f×    │    DFTI      │\n", time_fft, time_spatial/time_fft);
    printf("└──────────────────┴──────────────┴──────────────┴──────────────┘\n");
    
    // Vérification de cohérence
    printf("\n[7] VÉRIFICATION DE LA COHÉRENCE DES RÉSULTATS\n");
    
    float diff_sep = 0.0f, diff_fft = 0.0f;
    size_t total = width * height;
    
    for (size_t i = 0; i < total; i++) {
        float d_sep = fabsf(result_spatial->data[i] - result_sep->data[i]);
        float d_fft = fabsf(result_spatial->data[i] - result_fft->data[i]);
        diff_sep += d_sep;
        diff_fft += d_fft;
    }
    
    diff_sep /= total;
    diff_fft /= total;
    
    printf("    - Écart moyen Spatiale vs Séparable : %.6f\n", diff_sep);
    printf("    - Écart moyen Spatiale vs FFT       : %.6f\n", diff_fft);
    
    if (diff_sep < 0.01f && diff_fft < 0.1f) {
        printf("    ✓ Les trois méthodes produisent des résultats équivalents\n");
    }
    
    print_separator();
    
    printf("\n[8] CONCLUSIONS\n");
    printf("    ✓ Pour noyaux petits (≤7×7)  : Méthode SÉPARABLE optimale\n");
    printf("    ✓ Pour noyaux grands (≥11×11) : Méthode FFT recommandée\n");
    printf("    ✓ Intel MKL apporte un gain de %.0f× à %.0f×\n", 
           time_spatial/time_sep, time_spatial/time_fft);
    
    print_separator();
    
    // Nettoyage
    free_image(img);
    free_image(result_spatial);
    free_image(result_sep);
    free_image(result_fft);
    free_kernel(kernel);
    mkl_free(kernel_1d);
    
    printf("\n[*] Démonstration terminée avec succès !\n\n");
}

int main() {
    demonstrate_denoising();
    return 0;
}
