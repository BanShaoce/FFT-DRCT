import cmath
import math
import unittest

from fft_drct import drct, fft, idrct, ifft


class FFTDRCTTests(unittest.TestCase):
    def test_fft_inverse_round_trip(self):
        values = [complex(1, 2), complex(-3, 4), complex(5, -6), complex(0, 1)]

        transformed = fft(values)
        restored = ifft(transformed)

        for expected, actual in zip(values, restored):
            self.assertAlmostEqual(expected.real, actual.real, places=9)
            self.assertAlmostEqual(expected.imag, actual.imag, places=9)

    def test_fft_matches_direct_dft(self):
        values = [1, 2, 3, 4]
        expected = []
        size = len(values)
        for k in range(size):
            expected.append(
                sum(values[n] * cmath.exp(-2j * math.pi * k * n / size) for n in range(size))
            )

        actual = fft(values)

        for exp, got in zip(expected, actual):
            self.assertAlmostEqual(exp.real, got.real, places=9)
            self.assertAlmostEqual(exp.imag, got.imag, places=9)

    def test_drct_matches_direct_dct_ii(self):
        values = [0.5, -1.0, 2.0, 3.5]
        size = len(values)
        expected = [
            sum(values[n] * math.cos(math.pi * (n + 0.5) * k / size) for n in range(size))
            for k in range(size)
        ]

        actual = drct(values)

        for exp, got in zip(expected, actual):
            self.assertAlmostEqual(exp, got, places=9)

    def test_drct_inverse_round_trip(self):
        values = [2.0, 0.0, -1.0, 1.0, 3.0, -2.0, 0.5, 4.0]

        transformed = drct(values)
        restored = idrct(transformed)

        for expected, actual in zip(values, restored):
            self.assertAlmostEqual(expected, actual, places=8)

    def test_fft_rejects_non_power_of_two(self):
        with self.assertRaises(ValueError):
            fft([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
