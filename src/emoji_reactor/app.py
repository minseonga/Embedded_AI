"""
Emoji Reactor powered by our ONNX hand-tracking pipeline (pruning/quantization)
plus keyword-based mode switching (emoji ↔ monkey) via optional Vosk ASR.

States:
- HANDS_UP      : any hand bbox starts above --raise-thresh
- SMILING       : mouth aspect ratio exceeds --smile-thresh (MediaPipe Face Mesh)
- AHA           : only index finger raised
- CONFUSED      : index fingertip near mouth
- FRUSTRATED    : two hands above --frustrated-thresh
- STRAIGHT_FACE : fallback

Run examples:
  python emoji_reactor_onnx.py --precision fp16 --prune 0.3
  python emoji_reactor_onnx.py --precision int8 --raise-thresh 0.25 --smile-thresh 0.35
Voice hotwords (if Vosk + model installed): say “emoji” or “monkey” to switch sets.
"""

import argparse
import os
import sys
import time
import threading
import shutil
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import queue
import sounddevice as sd
import traceback
import collections
import types

# Paths
ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ASSETS = ROOT / "assets"
EMOJI_DIR = ASSETS / "emojis"
AUDIO_DIR = ASSETS / "audio"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_tracking import HandTrackingPipeline, draw_landmarks, draw_detections  # noqa: E402

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
EMOJI_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

SMILE_LANDMARKS = {
    "left_corner": 291,
    "right_corner": 61,
    "upper_lip": 13,
    "lower_lip": 14,
}


def load_emoji_set(prefix, names):
    loaded = {}
    missing = []
    for state, filename in names.items():
        path = EMOJI_DIR / f"{prefix}{filename}"
        img = cv2.imread(str(path))
        if img is None:
            missing.append(str(path))
            continue
        loaded[state] = cv2.resize(img, EMOJI_SIZE)

    return loaded, missing


def load_emojis():
    # Filenames per mode (prefix "" or "monkey_")
    file_map = {
        "SMILING": "smile.jpg",
        "STRAIGHT_FACE": "plain.png",
        "HANDS_UP": "air.jpg",
        "AHA": "aha.png",
        "CONFUSED": "confused.png",
        "FRUSTRATED": "frustrated.png",
    }
    monkey_file_map = {k: f"monkey_{v}" if not v.startswith("monkey_") else v for k, v in file_map.items()}

    default_set, missing_default = load_emoji_set("", file_map)
    monkey_set, missing_monkey = load_emoji_set("", monkey_file_map)

    if missing_default:
        print(f"[WARN] Missing default emoji files: {', '.join(missing_default)}")
    if missing_monkey:
        print(f"[WARN] Missing monkey emoji files: {', '.join(missing_monkey)}")

    blank = np.zeros((EMOJI_SIZE[1], EMOJI_SIZE[0], 3), dtype=np.uint8)
    return {
        "default": default_set,
        "monkey": monkey_set,
        "blank": blank
    }


def patch_tensorflow_stub():
    """Work around broken tensorflow installs that lack Tensor attribute."""
    try:
        import tensorflow as tf  # type: ignore
        if not hasattr(tf, "Tensor"):
            class _DummyTensor:
                pass
            tf.Tensor = _DummyTensor  # type: ignore[attr-defined]
            tf.constant = lambda *a, **k: None  # type: ignore
            tf.convert_to_tensor = lambda *a, **k: None  # type: ignore
            print("[VOICE] Patched missing tensorflow.Tensor")
    except Exception:
        pass


def mouth_aspect_ratio(face_landmarks):
    lc = face_landmarks.landmark[SMILE_LANDMARKS["left_corner"]]
    rc = face_landmarks.landmark[SMILE_LANDMARKS["right_corner"]]
    ul = face_landmarks.landmark[SMILE_LANDMARKS["upper_lip"]]
    ll = face_landmarks.landmark[SMILE_LANDMARKS["lower_lip"]]

    mouth_w = ((rc.x - lc.x) ** 2 + (rc.y - lc.y) ** 2) ** 0.5
    mouth_h = ((ll.x - ul.x) ** 2 + (ll.y - ul.y) ** 2) ** 0.5
    if mouth_w <= 0:
        return 0.0
    return mouth_h / mouth_w


def is_hand_up(det, scale, pad, frame_h, raise_thresh):
    """Check if the top of the detection box is above the threshold fraction of frame height."""
    y1 = det[0] * scale * 256 - pad[0]
    return (y1 / frame_h) < raise_thresh


def is_index_up_only(lm, frame_h):
    """
    Detect a loose "index up" gesture:
    - index tip higher than its pip a bit
    - index tip higher than other fingertips by a small delta
    """
    idx_tip = lm[8, 1]
    idx_pip = lm[6, 1]
    if idx_tip >= idx_pip - 0.003 * frame_h:
        return False

    other_tips = [lm[4, 1], lm[12, 1], lm[16, 1], lm[20, 1]]
    min_other = min(other_tips)
    # require index clearly highest
    if idx_tip >= min_other - 0.006 * frame_h:
        return False
    return True


def is_confused(lm, mouth_center, frame_h, thresh_ratio):
    if mouth_center is None:
        return False
    idx_tip = lm[8, :2]
    dist = np.linalg.norm(idx_tip - mouth_center)
    return dist / frame_h < thresh_ratio


def count_hands_up_wrist(hand_landmarks, frame_h, thresh):
    """Count hands whose wrist is above a fraction of frame height."""
    count = 0
    for lm in hand_landmarks:
        if lm[0, 1] / frame_h < thresh:
            count += 1
    return count


def draw_face_landmarks(frame, face_landmarks):
    """No-op face overlay (disabled)."""
    return


def draw_mic_meter(frame, level, origin=(10, 140), size=(180, 12)):
    """Draw a simple mic-level bar."""
    x, y = origin
    w, h = size
    level = max(0.0, min(1.0, level * 3.0))  # amplify a bit
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)
    cv2.rectangle(frame, (x, y), (x + int(w * level), y + h), (0, 200, 255), -1)


def resolve_input_device(input_device, fallback_rate=16000):
    """Resolve sounddevice input device index and preferred samplerate."""
    # Coerce numeric strings to int
    dev = input_device
    if isinstance(dev, str) and dev.isdigit():
        dev = int(dev)
    try:
        info = sd.query_devices(dev or None, 'input')
        dev_idx = info['index']
        rate = int(info['default_samplerate']) if info['default_samplerate'] > 0 else fallback_rate
        return dev_idx, rate
    except Exception as e:
        print(f"[VOICE] resolve_input_device fallback ({dev}): {e}")
        return dev, fallback_rate


def draw_voice_wave(frame, samples, origin=(10, 160), size=(200, 50), color=(0, 200, 255)):
    """Draw an oscilloscope-style waveform from mic levels."""
    if not samples:
        return
    x0, y0 = origin
    w, h = size
    pts = []
    n = len(samples)
    for i, s in enumerate(samples):
        x = x0 + int(w * i / max(1, n - 1))
        y = y0 + h // 2 - int(s * h)
        pts.append((x, y))
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)


class VoiceKeywordListener(threading.Thread):
    """Lightweight Vosk-based keyword listener (emoji/monkey). Optional."""

    def __init__(self, model_path=None, keywords=("emoji", "monkey"), sample_rate=16000, input_device=None):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.keywords = keywords
        self.sample_rate = sample_rate
        self.input_device = input_device
        self._last = None
        self._running = True
        self._ready = threading.Event()
        self.level = 0.0
        self.error = None
        self.device_info = None
        self.wave = collections.deque(maxlen=200)

    def stop(self):
        self._running = False
        # Let the audio callback exit promptly
        self._ready.set()

    def get_keyword(self):
        kw = self._last
        self._last = None
        return kw

    @property
    def is_ready(self):
        return self._ready.is_set()

    def run(self):
        try:
            import vosk
            import sounddevice as sd
        except ImportError:
            print("[VOICE] sounddevice/vosk not installed; voice control disabled")
            return

        model_dir = self.model_path or os.environ.get("VOSK_MODEL_PATH", "")
        if not model_dir or not os.path.isdir(model_dir):
            print("[VOICE] VOSK model path missing; set --vosk-model or VOSK_MODEL_PATH")
            return

        try:
            model = vosk.Model(model_dir)
            rec = vosk.KaldiRecognizer(model, self.sample_rate)
        except Exception as e:
            print(f"[VOICE] Failed to load Vosk model: {e}")
            return

        self._ready.set()
        print("[VOICE] Listening for: " + ", ".join(self.keywords))

        dev_idx, sr = resolve_input_device(self.input_device, self.sample_rate)
        try:
            self.device_info = sd.query_devices(dev_idx, 'input')
            print(f"[VOICE] Vosk using device {dev_idx}: {self.device_info['name']}, sr={sr}")
        except Exception:
            pass
        def callback(indata, frames, time_info, status):
            if not self._running:
                raise sd.CallbackStop
            # mic level (RMS)
            arr = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0
            self.level = 0.8 * self.level + 0.2 * rms
            self.wave.append(self.level)
            if rec.AcceptWaveform(bytes(indata)):
                res = rec.Result()
                if "text" in res:
                    txt = res.split('"')[-2].lower()
                    for k in self.keywords:
                        if k in txt:
                            self._last = k

        try:
            with sd.RawInputStream(samplerate=sr, blocksize=2048,
                                   dtype='int16', channels=1, callback=callback,
                                   device=dev_idx):
                while self._running:
                    time.sleep(0.1)
        except Exception as e:
            self.error = f"Audio stream error: {e}"
            print(f"[VOICE] Audio stream error: {e}")


class HFKeywordListener(threading.Thread):
    """Keyword listener using Hugging Face ASR models."""

    def __init__(self, model_id, keywords=("emoji", "monkey"), sample_rate=16000, chunk_seconds=4, device="cpu", base_repo="openai/whisper-tiny", input_device=None):
        super().__init__(daemon=True)
        self.model_id = model_id
        self.keywords = keywords
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.device = device
        self.base_repo = base_repo
        self.input_device = input_device
        self._last = None
        self._running = True
        self._ready = threading.Event()
        self._buffer = bytearray()
        self.error = None
        self.level = 0.0
        self.device_info = None
        self.wave = collections.deque(maxlen=200)

    def stop(self):
        self._running = False
        self._ready.set()

    def get_keyword(self):
        kw = self._last
        self._last = None
        return kw

    @property
    def is_ready(self):
        return self._ready.is_set()

    def run(self):
        # Force transformers to avoid TensorFlow/Flax imports
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
        patch_tensorflow_stub()

        try:
            import sounddevice as sd
        except ImportError:
            self.error = "sounddevice not installed"
            print("[VOICE] sounddevice not installed; HF voice disabled")
            self._ready.set()
            return
        try:
            import torch
            import transformers
            from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperConfig
            print(f"[VOICE] HF import ok: torch {torch.__version__}, transformers {transformers.__version__}")
        except ImportError as e:
            self.error = f"transformers/torch import failed: {e}"
            print(f"[VOICE] transformers/torch import failed: {e}")
            print("[VOICE] sys.path =", sys.path)
            self._ready.set()
            return

        model_id = self.model_id or self.base_repo
        try:
            torch_dtype = torch.float16 if torch.cuda.is_available() and self.device != "cpu" else None
            device_idx = 0 if self.device != "cpu" else -1

            processor = WhisperProcessor.from_pretrained(self.base_repo)
            if Path(model_id).is_file() and model_id.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                except ImportError:
                    self.error = "safetensors not installed"
                    print("[VOICE] safetensors not installed; install to use local .safetensors")
                    self._ready.set()
                    return
                print(f"[VOICE] Loading local weights from {model_id} with base {self.base_repo}")
                state = load_file(model_id, device="cpu")
                config = WhisperConfig.from_pretrained(self.base_repo)
                model = WhisperForConditionalGeneration(config)
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"[VOICE] Whisper state dict missing:{len(missing)} unexpected:{len(unexpected)}")
                if torch_dtype:
                    model = model.to(dtype=torch_dtype)
            else:
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch_dtype
                )
            if self.device != "cpu":
                model = model.to(self.device)
            model.eval()
        except Exception as e:
            self.error = f"Failed to load HF model {model_id}: {e}"
            print(f"[VOICE] Failed to load HF model {model_id}: {e}")
            self._ready.set()
            return

        self._ready.set()
        print(f"[VOICE] HF ASR ready: {model_id} (hotwords: {', '.join(self.keywords)})")

        dev_idx, sr = resolve_input_device(self.input_device, self.sample_rate)
        try:
            self.device_info = sd.query_devices(dev_idx, 'input')
            print(f"[VOICE] HF using device {dev_idx}: {self.device_info['name']}, sr={sr}")
        except Exception:
            pass
        sr_target = 16000
        chunk_bytes = int(sr_target * self.chunk_seconds * 2)  # int16 bytes

        def callback(indata, frames, time_info, status):
            if not self._running:
                raise sd.CallbackStop
            arr = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0
            self.level = 0.8 * self.level + 0.2 * rms
            self.wave.append(self.level)
            self._buffer.extend(bytes(indata))

        try:
            with sd.RawInputStream(samplerate=sr, blocksize=2048,
                                   dtype='int16', channels=1, callback=callback,
                                   device=dev_idx):
                while self._running:
                    if len(self._buffer) >= chunk_bytes:
                        data = np.frombuffer(self._buffer[:chunk_bytes], dtype=np.int16).astype(np.float32) / 32768.0
                        del self._buffer[:chunk_bytes]
                        try:
                            # Resample if needed
                            if sr != sr_target:
                                import librosa
                                data_16k = librosa.resample(data, orig_sr=sr, target_sr=sr_target)
                            else:
                                data_16k = data
                            inputs = processor(data_16k, sampling_rate=sr_target, return_tensors="pt")
                            if self.device != "cpu":
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            with torch.no_grad():
                                generated_ids = model.generate(inputs["input_features"])
                            txt = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()
                            for k in self.keywords:
                                if k in txt:
                                    self._last = k
                                    break
                        except Exception as e:
                            self.error = f"ASR error: {e}"
                            print(f"[VOICE] ASR error: {e}")
                    time.sleep(0.1)
        except Exception as e:
            self.error = f"Audio stream error: {e}"
            print(f"[VOICE] Audio stream error: {e}")


class BackgroundMusic(threading.Thread):
    """Loop yessir.mp3 using system players (afplay on mac, ffplay if available)."""

    def __init__(self, path, enabled=True):
        super().__init__(daemon=True)
        self.path = path
        self.enabled = enabled
        self._running = True
        self._ready = threading.Event()
        self._proc = None
        self._ok = False

    def stop(self):
        self._running = False
        self._ready.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    @property
    def is_ready(self):
        return self._ready.is_set() and self._ok

    def _player_cmd(self):
        # macOS built-in
        if sys.platform == "darwin" and shutil.which("afplay"):
            return ["afplay", self.path]
        # ffplay (ffmpeg) if available
        if shutil.which("ffplay"):
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.path]
        return None

    def run(self):
        if not self.enabled:
            return
        if not os.path.isfile(self.path):
            print(f"[MUSIC] File not found: {self.path}")
            return

        cmd = self._player_cmd()
        if not cmd:
            print("[MUSIC] No supported system player found (needs afplay or ffplay)")
            return

        print(f"[MUSIC] Playing on loop: {os.path.basename(self.path)} via {cmd[0]}")
        self._ok = True
        self._ready.set()

        while self._running:
            try:
                self._proc = subprocess.Popen(cmd)
                self._proc.wait()
            except Exception as e:
                print(f"[MUSIC] Playback error: {e}")
                break

        # cleanup
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Emoji Reactor with ONNX hand tracking.")
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16',
                        help='Model precision (fp16 recommended on Jetson/TensorRT).')
    parser.add_argument('--prune', type=float, default=0.0, help='Pruning ratio (0-1).')
    parser.add_argument('--camera', type=int, default=0, help='Camera index.')
    parser.add_argument('--raise-thresh', type=float, default=0.25,
                        help='Fraction of frame height; bbox top above this = HANDS_UP.')
    parser.add_argument('--smile-thresh', type=float, default=0.35,
                        help='Mouth aspect ratio threshold for SMILING.')
    parser.add_argument('--confused-thresh', type=float, default=0.08,
                        help='Index fingertip distance to mouth (fraction of frame height).')
    parser.add_argument('--frustrated-thresh', type=float, default=0.45,
                        help='Fraction of frame height; two wrists above this = FRUSTRATED (higher is easier).')
    parser.add_argument('--no-mirror', action='store_true', help='Disable horizontal flip.')
    parser.add_argument('--no-voice', action='store_true', help='Disable voice hotword switching.')
    parser.add_argument('--voice-backend', choices=['auto', 'vosk', 'hf'], default='auto',
                        help='Choose voice backend (auto prefers HF if provided).')
    parser.add_argument('--vosk-model', type=str, default=os.environ.get("VOSK_MODEL_PATH", ""),
                        help='Path to Vosk model directory (optional).')
    parser.add_argument('--hf-asr-model', type=str, default="",
                        help='Hugging Face ASR model id or local .safetensors path (whisper).')
    parser.add_argument('--hf-asr-base', type=str, default="openai/whisper-tiny",
                        help='Base repo to load config/tokenizer when using local weights.')
    parser.add_argument('--hf-device', type=str, default="cpu", help='Device for HF ASR (cpu or cuda).')
    parser.add_argument('--hf-chunk', type=int, default=4, help='Seconds per ASR chunk for HF backend.')
    parser.add_argument('--no-face', action='store_true', help='Disable drawing face landmarks.')
    parser.add_argument('--no-music', action='store_true', help='Disable background music.')
    parser.add_argument('--input-device', type=str, default=None,
                        help='sounddevice input device name or index for mic (e.g., 0 or "MacBook Pro Microphone").')
    args = parser.parse_args()

    emoji_sets = load_emojis()

    pipeline = HandTrackingPipeline(prune_ratio=args.prune, precision=args.precision)
    pipeline.print_stats()

    music_path = AUDIO_DIR / "yessir.mp3"
    music = None
    if not args.no_music:
        music = BackgroundMusic(str(music_path), enabled=True)
        music.start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Reactor View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.resizeWindow('Reactor View', WINDOW_WIDTH * 2, WINDOW_HEIGHT)

    fps_hist = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

    # Optional voice hotword listener
    listener = None
    voice_ready_notified = False
    last_voice_ts = 0.0
    voice_status = "Voice: off" if args.no_voice else "Voice: starting..."
    voice_backend = None
    # Auto-detect local whisper weights if available
    default_whisper = ROOT / "whisper.safetensors"
    if not args.hf_asr_model and default_whisper.exists():
        args.hf_asr_model = str(default_whisper)
        args.voice_backend = 'hf'

    if not args.no_voice:
        backend = args.voice_backend
        if backend == 'auto':
            backend = 'hf' if args.hf_asr_model else 'vosk'
        voice_backend = backend
        if backend == 'hf' and args.hf_asr_model:
            listener = HFKeywordListener(
                model_id=args.hf_asr_model,
                keywords=("emoji", "monkey"),
                sample_rate=16000,
                chunk_seconds=args.hf_chunk,
                device=args.hf_device,
                base_repo=args.hf_asr_base,
                input_device=args.input_device
            )
        elif backend == 'vosk':
            listener = VoiceKeywordListener(model_path=args.vosk_model, input_device=args.input_device)

        if listener:
            listener.start()
        else:
            voice_status = f"Voice[{voice_backend}]: not configured"

    active_mode = "default"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not args.no_mirror:
            frame = frame[:, ::-1].copy()

        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        h, w = frame.shape[:2]

        start = time.time()
        hand_landmarks, detections, scale, pad = pipeline.process_frame(frame)
        dt = time.time() - start

        fps = 1.0 / dt if dt > 0 else 0.0
        fps_hist.append(fps)
        if len(fps_hist) > 30:
            fps_hist.pop(0)
        avg_fps = float(np.mean(fps_hist))

        # Face mesh for smile
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_res = face_mesh.process(rgb)
        mar = 0.0
        mouth_center = None
        if face_res.multi_face_landmarks:
            mar = mouth_aspect_ratio(face_res.multi_face_landmarks[0])
            ul = face_res.multi_face_landmarks[0].landmark[SMILE_LANDMARKS["upper_lip"]]
            ll = face_res.multi_face_landmarks[0].landmark[SMILE_LANDMARKS["lower_lip"]]
            mouth_center = np.array([(ul.x + ll.x) * 0.5 * w, (ul.y + ll.y) * 0.5 * h])
            if not args.no_face:
                draw_face_landmarks(frame, face_res.multi_face_landmarks[0])

        # State decision (mode-specific)
        if active_mode == "monkey":
            state = "STRAIGHT_FACE"
            frustrated = count_hands_up_wrist(hand_landmarks, h, args.frustrated_thresh) >= 2
            conf_flag = any(is_confused(lm, mouth_center, h, args.confused_thresh) for lm in hand_landmarks)
            aha_flag = any(is_index_up_only(lm, h) for lm in hand_landmarks) and not conf_flag

            if frustrated:
                state = "FRUSTRATED"
            elif aha_flag:
                state = "AHA"
            elif conf_flag:
                state = "CONFUSED"
        else:  # emoji/default mode
            state = "STRAIGHT_FACE"
            if len(detections) > 0 and any(is_hand_up(det, scale, pad, h, args.raise_thresh) for det in detections):
                state = "HANDS_UP"
            elif mar > args.smile_thresh:
                state = "SMILING"

        # Voice mode switch
        if listener:
            kw = listener.get_keyword()
            if kw in ("emoji", "monkey"):
                active_mode = "monkey" if kw == "monkey" else "default"
                voice_status = f"Voice[{voice_backend}]: {kw}"
                last_voice_ts = time.time()
                print(f"[VOICE] switched to {active_mode} mode")
            if listener.error:
                voice_status = f"Voice[{voice_backend}]: {listener.error}"
            elif listener.is_ready and not voice_ready_notified:
                print("[VOICE] Ready (hotwords: emoji, monkey)")
                voice_status = f"Voice[{voice_backend}]: ready"
                voice_ready_notified = True

        if active_mode == "monkey":
            emoji = emoji_sets.get("monkey", {}).get(state)
            if emoji is None:
                emoji = emoji_sets["blank"]  # stay in monkey mode; no cross-mode fallback
        else:
            emoji = emoji_sets["default"].get(state)
            if emoji is None:
                emoji = emoji_sets["blank"]
        emoji_char = {
            "HANDS_UP": "🙌",
            "SMILING": "😊",
            "STRAIGHT_FACE": "😐",
            "AHA": "💡",
            "CONFUSED": "🤔",
            "FRUSTRATED": "😤",
        }.get(state, "❓")

        vis = frame.copy()
        for lm in hand_landmarks:
            draw_landmarks(vis, lm)
        if len(detections) > 0:
            draw_detections(vis, detections, scale, pad)

        cv2.putText(vis, f"STATE: {state} {emoji_char}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"FPS {avg_fps:.1f} | {args.precision.upper()} | prune {args.prune*100:.0f}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Mode: {active_mode}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        # Voice overlay (show last hotword for 3s) + mic bar
        if not args.no_voice:
            if time.time() - last_voice_ts > 3 and voice_ready_notified:
                voice_status = f"Voice[{voice_backend}]: ready"
            voice_level = getattr(listener, "level", 0.0) if listener else 0.0
            voice_wave = list(getattr(listener, "wave", []))
            cv2.putText(vis, voice_status, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 220, 255), 2)
            draw_mic_meter(vis, voice_level, origin=(10, 140), size=(180, 12))
            draw_voice_wave(vis, voice_wave, origin=(10, 170), size=(220, 50), color=(0, 255, 180))
        cv2.putText(vis, 'Press "q" or ESC to quit', (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow('Camera Feed', vis)
        cv2.imshow('Emoji Output', emoji)
        combined = np.hstack((vis, emoji))
        cv2.imshow('Reactor View', combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('m'):
            active_mode = "monkey"
            print("[KEY] switched to monkey mode")
        if key == ord('e'):
            active_mode = "default"
            print("[KEY] switched to emoji mode")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    if listener:
        listener.stop()
    if music:
        music.stop()


if __name__ == "__main__":
    main()
