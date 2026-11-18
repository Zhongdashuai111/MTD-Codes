# Source attribution and modifications

The files in `vendor/` originate from the Paiva et al. repository
<https://github.com/thalespaiva/minal_mlkem>, commit
`6b12ae4fa24c04fe756386cb8d7e7d88d283d1a9`.
Their Apache-2.0 license is retained in `vendor/LICENSE`.

`vendor/common.h` was adapted to allow command-line overrides of
`N_PRECISION_BITS` and `N_SUPPORT_DIST_2D` using `#ifndef` guards.
The other bundled vendor files retain the author implementation.

`fft2d_custom.c` is an adapter prepared for the MTD-code DFR comparison.
It uses the author's base-distribution routines and multiprecision FFT,
with the two specified codebooks, an explicit toroidal event check,
periodic-neighbor halfspace summation, a 2048-period grid, separate
coordinate-noise transforms, and JSON output. It is not an original
Paiva-repository file.
