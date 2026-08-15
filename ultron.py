"""A small offline voice assistant built with Vosk, Piper, and sounddevice."""

import json
import queue
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice
from vosk import KaldiRecognizer, Model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VOSK_MODEL_PATH = Path(r"C:\Ultron\models\vosk-model-small-en-us-0.15")
PIPER_MODEL_PATH = Path(r"C:\Piper\en_US-ryan-medium.onnx")

# Set this to a device number or part of a device name to override automatic
# microphone selection. Leave as None to use the system's default microphone.
MICROPHONE_DEVICE = None

VOSK_SAMPLE_RATE = 16_000
BLOCK_DURATION_SECONDS = 0.1
# Raise this only if the live partial transcript remains empty while speaking.
# Values above 4 can distort loud speech.
MICROPHONE_GAIN = 2.0


def require_file(path: Path, description: str) -> None:
    """Stop with a useful message when a required model is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found: {path}")


def select_microphone(preferred_device=None) -> tuple[int, dict]:
    """Return an input-capable device, preferring the requested/default device."""
    devices = sd.query_devices()

    if preferred_device is not None:
        if isinstance(preferred_device, int):
            candidate = preferred_device
        else:
            needle = str(preferred_device).lower()
            candidate = next(
                (index for index, device in enumerate(devices)
                 if needle in device["name"].lower() and device["max_input_channels"] > 0),
                None,
            )
        if candidate is None:
            raise ValueError(f"No input device matches: {preferred_device!r}")
        device = sd.query_devices(candidate)
        if device["max_input_channels"] < 1:
            raise ValueError(f"Selected device is not an input device: {device['name']}")
        return candidate, device

    default_input, _ = sd.default.device
    if default_input is not None and default_input >= 0:
        device = sd.query_devices(default_input)
        if device["max_input_channels"] > 0:
            return default_input, device

    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            return index, device

    raise RuntimeError("No microphone/input device was found.")


def audio_to_pcm(audio: np.ndarray) -> bytes:
    """Convert mono microphone samples to Vosk's 16-bit PCM format."""
    samples = np.clip(audio[:, 0] * MICROPHONE_GAIN, -1.0, 1.0)
    return (samples * 32767).astype(np.int16).tobytes()


def speak(voice: PiperVoice, text: str) -> None:
    print(f"Ultron: {text}")
    for chunk in voice.synthesize(text):
        sd.play(chunk.audio_int16_array, chunk.sample_rate)
        sd.wait()


def ultron_response(text: str) -> tuple[str, bool]:
    """Return Ultron's reply and whether the program should exit."""
    command = text.lower().strip()

    if command in {"exit", "shutdown", "quit", "goodbye"}:
        return "Shutting down. Goodbye.", True
    if "hello" in command or command == "hi":
        return "Hello. Systems are online. How can I assist you?", False
    if "who are you" in command:
        return "I am Ultron, your personal AI assistant.", False
    if "how are you" in command:
        return "All systems are operating normally.", False
    if "your name" in command:
        return "My designation is Ultron.", False

    return "I heard you. My intelligence module is ready for expansion.", False


def main() -> None:
    require_file(VOSK_MODEL_PATH, "Vosk model")
    require_file(PIPER_MODEL_PATH, "Piper voice model")

    print("Loading Ultron voice...")
    voice = PiperVoice.load(str(PIPER_MODEL_PATH))
    print("Loading speech recognition...")
    recognizer = KaldiRecognizer(Model(str(VOSK_MODEL_PATH)), VOSK_SAMPLE_RATE)

    microphone_index, microphone = select_microphone(MICROPHONE_DEVICE)
    # The device supports 16 kHz, which is exactly the format required by Vosk.
    # Capturing at that rate avoids conversion artifacts and reduces latency.
    microphone_rate = VOSK_SAMPLE_RATE
    blocksize = int(microphone_rate * BLOCK_DURATION_SECONDS)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=20)

    def audio_callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        try:
            audio_queue.put_nowait(indata.copy())
        except queue.Full:
            # Dropping old speech is better than letting recognition lag behind.
            pass

    print("\n==============================")
    print("       ULTRON ONLINE")
    print("==============================")
    print(f"Microphone: {microphone['name']} (device {microphone_index})")
    print("Speak clearly, then pause briefly. Partial text appears while listening.")
    print("Say 'exit' or 'shutdown' to quit.\n")

    with sd.InputStream(
        device=microphone_index,
        channels=1,
        samplerate=microphone_rate,
        blocksize=blocksize,
        dtype="float32",
        callback=audio_callback,
    ):
        last_partial = ""
        while True:
            pcm = audio_to_pcm(audio_queue.get())
            if not recognizer.AcceptWaveform(pcm):
                partial = json.loads(recognizer.PartialResult()).get("partial", "")
                if partial and partial != last_partial:
                    print(f"Listening: {partial}     ", end="\r", flush=True)
                    last_partial = partial
                continue

            result = json.loads(recognizer.Result())
            last_partial = ""
            text = result.get("text", "").strip()
            if not text:
                continue

            print(f"You: {text}")
            reply, should_exit = ultron_response(text)
            speak(voice, reply)
            if should_exit:
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUltron stopped.")
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
