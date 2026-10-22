"""
Real-Time Acoustic Defense Masker
Generates adaptive noise that neutralizes keystroke spectral signatures.
"""

import numpy as np
from scipy.signal import butter, lfilter, welch
from scipy.interpolate import interp1d
from typing import Optional


class AcousticDefenseMasker:
    """
    Analyzes incoming audio for typing-like spectral signatures
    and generates anti-phase masking noise in real time.

    Defense strategies:
    1. broadband  — shaped pink noise matched to keyboard spectrum
    2. adaptive   — real-time spectral gap filling
    3. burst      — random high-amplitude noise bursts on key events
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        strategy: str = "adaptive",
        snr_target_db: float = -5.0,
        vad_threshold: float = 0.01,
    ):
        self.sr = sample_rate
        self.strategy = strategy
        self.snr_target_db = snr_target_db
        self.vad_threshold = vad_threshold
        self._noise_floor: Optional[np.ndarray] = None
        self._prev_frame_energy = 0.0

    def is_typing_event(self, frame: np.ndarray) -> bool:
        """
        Voice Activity Detection tuned for keyboard acoustics.
        Keys: short, impulsive, high-frequency burst (1-6 kHz).
        """
        energy = np.sqrt(np.mean(frame ** 2))
        # Compute high-band energy fraction
        n = len(frame)
        spec = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(n, 1.0 / self.sr)
        high_mask = freqs > 2000
        high_frac = spec[high_mask].sum() / (spec.sum() + 1e-9)

        is_active = energy > self.vad_threshold
        is_keyboard = high_frac > 0.35
        return is_active and is_keyboard

    def _generate_pink_noise(self, n: int) -> np.ndarray:
        """Approximate pink noise via filtered white noise."""
        white = np.random.randn(n + 2048)
        b, a = butter(4, 0.4, btype="low")
        pink = lfilter(b, a, white)[:n]
        return pink / (np.abs(pink).max() + 1e-9)

    def generate_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate anti-attack masking noise for one audio frame.
        Returns noise signal to be mixed with the environment.
        """
        n = len(frame)

        if self.strategy == "broadband":
            noise = self._generate_pink_noise(n)
            scale = np.sqrt(np.mean(frame ** 2)) * 10 ** (-self.snr_target_db / 20)
            return noise * scale * 0.3

        elif self.strategy == "adaptive":
            # Spectral analysis of typing frame
            freqs, psd = welch(frame, fs=self.sr, nperseg=256)
            if self._noise_floor is None:
                self._noise_floor = psd.copy()

            # Amplify frequencies where keyboard energy peaks
            boost = np.maximum(psd / (self._noise_floor + 1e-9), 1.0)
            boost = np.clip(boost, 1, 10)

            # Generate shaped noise
            noise = np.fft.irfft(
                np.random.randn(n) * boost.mean(), n=n
            ).astype(np.float32)
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * psd
            return noise * 0.15

        elif self.strategy == "burst":
            if self.is_typing_event(frame):
                amplitude = np.random.uniform(0.3, 0.8)
                return np.random.randn(n).astype(np.float32) * amplitude
            return np.zeros(n, dtype=np.float32)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def process_stream_chunk(self, chunk: np.ndarray, chunk_size: int = 512) -> np.ndarray:
        """Process a continuous audio stream in overlapping frames."""
        masked = []
        for i in range(0, len(chunk) - chunk_size, chunk_size):
            frame = chunk[i: i + chunk_size]
            mask = self.generate_mask(frame)
            masked.append(frame + mask)

        if not masked:
            return chunk
        result = np.concatenate(masked)
        # Append any remaining tail
        remainder = len(chunk) % chunk_size
        if remainder:
            result = np.concatenate([result, chunk[-remainder:]])
        return result
