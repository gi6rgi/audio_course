import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def get_audio_peaks(y: np.ndarray, sr: int, height: float = 0.0) -> np.ndarray:
    time = np.linspace(0, len(y) / sr, len(y))
    peaks, _ = find_peaks(y, distance=100, height=height, prominence=0.01)
    peaks_time = time[peaks]
    return peaks_time


def plot_period(y: np.ndarray, sr: int) -> None:
    peaks_time = get_audio_peaks(y, sr, 0.4)
    if len(peaks_time) >= 2:
        T_start = peaks_time[0]
        T_end = peaks_time[1]

        plt.axvline(x=T_start, color="red", linestyle="--")
        plt.axvline(x=T_end, color="red", linestyle="--")

        arrow_y = np.max(np.abs(y)) * 0.8
        plt.annotate(
            "",
            xy=(T_end, arrow_y),
            xytext=(T_start, arrow_y),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
        )
        plt.text(
            (T_start + T_end) / 2,
            arrow_y + 0.05 * np.max(np.abs(y)),
            f"Период колебаний, T",
            color="red",
            ha="center",
        )


def plot_audio_signal(
    y: np.ndarray,
    sr: int,
    title: str,
    color: str = "blue",
    max_y: float = None,
    show_signal_period: bool = False,
) -> None:
    t = np.linspace(0, len(y) / sr, num=len(y))
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, alpha=0.7, color=color)
    plt.title(title)
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")

    if max_y is not None:
        plt.ylim(bottom=-max_y, top=max_y)

    if show_signal_period:
        plot_period(y=y, sr=sr)

    plt.grid(True)

    plt.show()


def get_audio_peaks(y: np.ndarray, sr: int, height: float = 0.0) -> np.ndarray:
    time = np.linspace(0, len(y) / sr, len(y))
    peaks, _ = find_peaks(y, distance=100, height=height, prominence=0.01)
    peaks_time = time[peaks]
    return peaks_time
