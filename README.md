# FFT-DRCT

Minimal Python implementation of:

- power-of-two 1D FFT / inverse FFT
- DRCT (implemented with a DCT-II style FFT formulation)
- inverse DRCT

## Quick start

```python
from fft_drct import drct, idrct

values = [1.0, 2.0, 3.0, 4.0]
coefficients = drct(values)
restored = idrct(coefficients)
```