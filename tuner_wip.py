import queue

import librosa
import numpy as np
import sounddevice as sd

SR = 44100
BLOCKSIZE = 512
BUFFERSIZE = SR * 2

audio_buffer = queue.Queue()


def predict_note(y: np.ndarray) -> str:
    N = len(y)
    fft_values = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, 1 / SR)

    half_N = N // 2
    freqs_pos = freqs[:half_N]
    fft_pos = fft_values[:half_N]

    amplitude = np.abs(fft_pos)

    # Найти индекс максимальной амплитуды
    max_index = np.argmax(amplitude)
    peak_freq = freqs_pos[max_index]

    note = librosa.hz_to_note(peak_freq)
    print(f"Detected freq: {peak_freq:.2f} Hz -> Note: {note}")
    return note


def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time) -> None:
    data = indata.flatten()
    audio_buffer.put(data)


stream = sd.InputStream(callback=callback, samplerate=SR, blocksize=BLOCKSIZE)
stream.start()

buffer = []
buffer_length = 0
while True:
    try:
        data = audio_buffer.get_nowait()
        buffer_length += len(data)
        buffer.append(data)

        if buffer_length >= BUFFERSIZE:
            collected_data_for_prediction = np.concatenate(buffer)
            predict_note(y=collected_data_for_prediction)

            buffer_length = 0
            # buffer = buffer[len(buffer) // 2:]
            buffer = []
    except queue.Empty:
        pass
    except KeyboardInterrupt:
        stream.stop()
        break
