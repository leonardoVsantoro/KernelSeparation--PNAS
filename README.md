# Kernel Embeddings and the Separation of Measure Phenomenon

Code for the paper **"Kernel Embeddings and the Separation of Measure Phenomenon"**
by Leonardo V. Santoro, Kartik G. Waghmare, and Victor M. Panaretos.

## Structure

```
code/
  src/
    models.py           # AR1 covariance model
    stats.py            # KLR and MMD statistics, Monte Carlo runner
    plot.py             # Result plotting (res_plot)
    figure_generator.py # All figure functions (figure_1a ... figure_3)
  figures.py            # Reproduce all figures (Figures 1–3)
  mc.py                 # Reproduce Figure 3 only (Monte Carlo experiment)
fig/                    # Output directory for saved figures
```

## Usage

From the code directory:

```bash
# Reproduce all figures
python figures.py

# Reproduce Figure 3 only (Monte Carlo power comparison)
python mc.py
```

Figures are saved to `../fig/`.
