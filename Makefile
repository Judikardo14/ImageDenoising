# Makefile pour la démonstration de débruitage d'images
# Compilation avec Intel MKL

# Compilateur
CC = gcc

# Options de compilation
CFLAGS = -O3 -Wall -std=c11

# Chemins Intel MKL (à adapter selon votre installation)
# Pour Linux avec Intel oneAPI:
MKL_ROOT = /opt/intel/oneapi/mkl/latest
MKL_INCLUDE = -I$(MKL_ROOT)/include
MKL_LIBS = -L$(MKL_ROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# Alternative pour linking dynamique (recommandé)
LDFLAGS = -Wl,--start-group $(MKL_ROOT)/lib/intel64/libmkl_intel_lp64.a \
          $(MKL_ROOT)/lib/intel64/libmkl_sequential.a \
          $(MKL_ROOT)/lib/intel64/libmkl_core.a \
          -Wl,--end-group -lpthread -lm -ldl

# Cibles
all: demo

demo: demo_debruitage.c
	@echo "=== Compilation du programme de démonstration ==="
	$(CC) $(CFLAGS) $(MKL_INCLUDE) -o demo_debruitage demo_debruitage.c $(MKL_LIBS)
	@echo "✓ Compilation réussie !"
	@echo "Exécutez avec : ./demo_debruitage"

# Version simplifiée (si MKL est dans les paths système)
demo_simple: demo_debruitage.c
	$(CC) $(CFLAGS) -o demo_debruitage demo_debruitage.c -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

# Nettoyage
clean:
	rm -f demo_debruitage *.o

# Test d'exécution
run: demo
	./demo_debruitage

# Afficher les informations MKL
info:
	@echo "=== Configuration Intel MKL ==="
	@echo "MKL_ROOT: $(MKL_ROOT)"
	@echo "Include:  $(MKL_INCLUDE)"
	@echo ""
	@echo "Pour vérifier l'installation MKL:"
	@echo "  ls $(MKL_ROOT)/lib/intel64/"

.PHONY: all clean run info demo_simple
