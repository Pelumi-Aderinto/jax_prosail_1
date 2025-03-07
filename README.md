# PROSAIL-JAX Python Bindings

**Adapted from the original NumPy/Torch versions by [Pelumi Aderinto/Instadeep].**

This repository contains **JAX-based** Python code for the PROSPECT(5, D and PRO) and SAIL (PROSAIL) leaf and canopy reflectance models. The original FORTRAN code was downloaded from [Jussieu](http://teledetection.ipgp.jussieu.fr/prosail/) and [jgomezdans](https://github.com/jgomezdans/prosail/tree/2.0.0alpha). We have reimplemented the functionality in Python + JAX. This code is useful for radiative transfer modeling in vegetation remote sensing, with automatic differentiation and GPU acceleration possibilities via JAX.

**Key references**:
- Original PROSPECT model: [Jacquemoud & Baret (1990), Jacquemoud et al. (2009)]
- Original SAIL model: [Verhoef (1984, 2007)]
- PROSAIL coupling: [Jacquemoud et al. (2009)]

---

## Installation

1. **Clone or download** this repository.
2. **Enter** the jax folder containing `setup.py`.
3. In your Python (3.12+) environment (preferrably create a virtual environment), run:
   ~~~bash
   pip install -e .
   ~~~
   This installs `prosail` in editable mode. You need:

   `jax, jaxlib, numpy, scipy` all detailed in the requirement.txt file in the jax folder

### Notes

- **No FORTRAN needed**: Unlike older versions, this reimplementation is purely in Python/JAX.
- **Double precision**: By default, JAX uses single-precision (float32). If you need float64, add:

  ~~~python
  import jax
  jax.config.update("jax_enable_x64", True)
  ~~~
  near the start of your script.

---

## Usage

You can **import** and call the main functions in Python:

~~~python
import prosail
import jax.numpy as jnp

# 1) Run PROSPECT alone:
wavelengths, reflectance, transmittance = prosail.run_prospect(
    n=2.1,  # Leaf structure
    cab=40, # Chlorophyll concentration, etc.
    car=10,
    cbrown=0.1,
    cw=0.015,
    cm=0.009,
    prospect_version="5"  # or "D" or "PRO" (Default is PRO)
)

# 2) Run 4SAIL alone:
canopy_refl = prosail.run_sail(
    refl=reflectance,
    trans=transmittance,
    lai=3.0,
    lidfa=-0.35,
    lidfb=-0.15,
    rsoil=1.0,
    psoil=0.3,
    hspot=0.1,
    tts=30.0,
    tto=10.0,
    psi=0.0,
    typelidf=1
)

# 3) Run PROSAIL end-to-end:
reflectance = prosail.run_prosail(
    n=1.5, cab=40, car=8, cbrown=0.0, cw=0.01, cm=0.009, cp=0.001, cbc=0.01,
    lai=3.0, lidfa=-0.35, lidfb=-0.15, rsoil=0.2, psoil=0.3, hspot=0.01,
    tts=30.0, tto=10.0, psi=0.0, typelidf=1
)
~~~

The typical wavelength range is `[400..2500]` nm in steps of 1 nm.

### Common Troubleshooting

- **Immutable arrays**: JAX does not allow in-place operations (e.g. `x[i] = y`). Use `x = x.at[i].set(y)` or `jnp.where(...)`.
- **Strings in jitted code**: If you pass a string to a jitted function (e.g., `prospect_version="5"`), you may need to remove `@jax.jit` or mark that argument as a static arg in your code.

---

## Parameters

Below are the key **PROSAIL** parameters and typical ranges:

| Parameter | Description of parameter         | Units   | Typical min | Typical max |
|-----------|----------------------------------|---------|------------|-------------|
| **N**     | Leaf structure parameter         | -       | 0.8        | 2.5         |
| **cab**   | Chlorophyll a+b concentration    | ug/cm2  | 0          | 80          |
| **car**   | Carotenoid concentration         | ug/cm2  | 0          | 20          |
| **cbrown**| Brown pigment                    | -       | 0          | 1           |
| **cw**    | Equivalent leaf water thickness  | cm      | 0          | 200         |
| **cm**    | Dry matter content               | g/cm2   | 0          | 200         |
| **lai**   | Leaf Area Index                  | -       | 0          | 10          |
| **lidfa** | Leaf angle distribution param A  | -       | -          | -           |
| **lidfb** | Leaf angle distribution param B  | -       | -          | -           |
| **psoil** | Dry/Wet soil factor              | -       | 0          | 1           |
| **rsoil** | Soil brightness factor           | -       | 0          | -           |
| **hspot** | Hotspot parameter                | -       | 0          | -           |
| **tts**   | Solar zenith angle               | deg     | 0          | 90          |
| **tto**   | Observer zenith angle            | deg     | 0          | 90          |
| **psi**   | Relative azimuth angle           | deg     | 0          | 360         |
| **typelidf** | Leaf angle distribution type  | Integer | -          | -           |

### Specifying the Leaf Angle Distribution

**`typelidf=1`** uses Verhoef’s two-parameter LIDF, where `(lidfa, lidfb)` control the average leaf slope and bimodality. Some typical combos:

| LIDF type     | `lidfa`  | `lidfb` |
|---------------|---------:|--------:|
| Planophile    |  1.0     | 0.0     |
| Erectophile   | -1.0     | 0.0     |
| Plagiophile   |  0.0     | -1.0    |
| Extremophile  |  0.0     |  1.0    |
| Spherical     | -0.35    | -0.15   |
| Uniform       |  0.0     |  0.0    |

**`typelidf=2`** means **ellipsoidal** distribution (Campbell’s parameterization), where `lidfa` stands for the mean leaf angle (0°=planophile, 90°=erectophile). `lidfb` is unused.

---

## Testing
We provide two test scripts:

1. test_prospect.py
2. test_prosail.py
   
They use pytest to verify that our JAX PROSAIL implementation matches expected outputs. From the project directory, run:

~~~bash
pytest tests/
~~~

### This will execute the test suite, where you can see some example usages and confirm that the code runs correctly.
---

## Contributing and Issues

1. **Pull requests** are welcome for bug fixes or enhancements.
2. **Issues**: If you encounter shape mismatch, JAX tracer errors, or other problems, please open an issue with sample code.

