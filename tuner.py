import queue
import threading

import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

SR = 44100
BLOCKSIZE = 512
BUFFERSIZE = SR * 5
TUNING_TOLERANCE = 1.0

raw_audio_data_queue = queue.Queue()
latest_freq = 0.0
target_freq = 0.0
latest_note = ""


def callback(data: np.ndarray, *args, **kwargs) -> None:
    raw_audio_data_queue.put_nowait(data.copy())


def predict_freq(y: np.ndarray) -> float | None:
    N = len(y)
    y = y * np.hanning(N)
    fft_values = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, 1 / SR)
    magnitude = np.abs(fft_values[: N // 2])
    freqs = freqs[: N // 2]
    max_index = np.argmax(magnitude)
    max_magnitude = magnitude[max_index]

    if max_magnitude > 50:
        return freqs[max_index]
    return None


def hz_to_nearest_note_freq(hz: float) -> tuple[str, np.ndarray]:
    note = librosa.hz_to_note(hz)
    freq = librosa.note_to_hz(note)
    return note, freq


def process_raw_audio_data_loop() -> None:
    global latest_freq, latest_note, target_freq

    buffer = []
    buffer_size = 0

    while True:
        try:
            data_chunk = raw_audio_data_queue.get_nowait()
            buffer.append(data_chunk.ravel())
            buffer_size = sum(chunk.size for chunk in buffer)

            if buffer_size >= BUFFERSIZE:
                y = np.concatenate(buffer)
                freq = predict_freq(y)

                # from scipy.io.wavfile import write
                # y = np.clip(y, -1.0, 1.0)

                # # Преобразуем float32 в int16
                # y_int16 = (y * 32767).astype(np.int16)

                # # Сохраняем
                # write("output.wav", SR, y_int16.reshape(-1, 1))

                if freq:
                    note, target = hz_to_nearest_note_freq(freq)
                    latest_freq = freq
                    latest_note = note
                    target_freq = target

                buffer = buffer[len(buffer) // 4 :]

        except queue.Empty:
            pass


def plot_tuner_data_loop() -> None:
    plt.ion()
    fig, ax = plt.subplots()
    while True:
        ax.clear()
        ax.set_xlim(target_freq - 10, target_freq + 10)
        ax.set_ylim(0, 1)
        ax.set_title(f"Target: {latest_note} ({target_freq:.2f} Hz)")
        ax.axvline(target_freq, color="blue", linestyle="--", label="Target")
        color = "green" if abs(latest_freq - target_freq) < TUNING_TOLERANCE else "red"
        ax.axvline(latest_freq, color=color, label=f"Detected: {latest_freq:.2f} Hz")
        ax.legend()
        plt.pause(0.3)


stream = sd.InputStream(
    callback=callback,
    blocksize=BLOCKSIZE,
    samplerate=SR,
    dtype='float32',
    channels=1,
)
audio_processing_thread = threading.Thread(
    target=process_raw_audio_data_loop, daemon=True
)

stream.start()
audio_processing_thread.start()
try:
    plot_tuner_data_loop()
except KeyboardInterrupt:
    stream.stop()
