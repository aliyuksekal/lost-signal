"""Narration + synthesized cinematic score (royalty-free, made here) -> final_audio.wav"""
import json, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt
SR = 24000
voice, _ = sf.read("/home/claude/work/narration_bm_george.wav")
tm = json.load(open("/home/claude/work/timing_bm_george.json"))
seg = tm["segments"]; T = tm["total"]; N = len(voice); T = N / SR; t = np.arange(N) / SR
rng = np.random.default_rng(7)

def env(a, b, fade=1.5):
    return np.clip((t - a) / fade, 0, 1) * np.clip((b - t) / fade, 0, 1)

def pad(freqs, amp):
    out = np.zeros(N)
    for f in freqs:
        for det in (-0.6, 0.0, 0.7):  # detuned layers = warm pad
            ph = rng.uniform(0, 2 * np.pi)
            out += np.sin(2 * np.pi * (f + det) * t + ph) + 0.25 * np.sin(2 * np.pi * 2 * (f + det) * t + ph)
    return out * amp / (len(freqs) * 3)

# chord progression (A minor feel): Am - F - C - G, slow, then hopeful Am-F-C at the end
chords = [([110, 164.8, 220, 261.6], 0, 10.3),        # wonder
          ([87.3, 130.8, 174.6, 220], 10.3, 19.8),     # doubt
          ([98, 146.8, 196, 233.1], 19.8, 30.2),        # darker (Gm)
          ([87.3, 130.8, 174.6, 207.7], 30.2, 41.9),    # lonely (Fm)
          ([130.8, 196, 261.6, 329.6], 41.9, T)]        # warm resolve (C)
music = np.zeros(N)
for fr, a, b in chords:
    music += pad(fr, 0.16) * env(a - 0.8, b + 0.8, 1.6)
music = sosfilt(butter(2, 1800, "low", fs=SR, output="sos"), music)
# soft piano-like plucks during the opening and the ending
def pluck(at, f, amp=0.12):
    i = int(at * SR); n = min(int(2.5 * SR), N - i)
    if n <= 0: return
    tt = np.arange(n) / SR
    music[i:i + n] += amp * np.exp(-tt / 0.7) * (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * 2 * f * tt) + 0.15 * np.sin(2 * np.pi * 3 * f * tt))
for k, at in enumerate(np.arange(0.4, 10, 1.2)):
    pluck(at, [440, 523.3, 659.3, 523.3][k % 4], 0.07)
for k, at in enumerate(np.arange(42.0, T - 1, 1.0)):
    pluck(at, [523.3, 659.3, 784, 659.3][k % 4], 0.07)
# power-down "whoomp" when lights go out, low drone + water rumble in subway
def hit(at, amp=0.5):
    i = int(at * SR); n = min(int(2.2 * SR), N - i); tt = np.arange(n) / SR
    music[i:i + n] += amp * np.sin(2 * np.pi * (70 - 30 * tt) * tt) * np.exp(-tt / 0.5)
for at in (seg[3]["start"] + 1.6, seg[3]["start"] + 3.2):
    hit(at, 0.35)
rumble = sosfilt(butter(2, [60, 400], "bandpass", fs=SR, output="sos"), rng.standard_normal(N)) * 0.05 * env(seg[4]["start"] - 0.3, seg[4]["end"] + 0.6, 0.8)
wind = sosfilt(butter(2, [300, 1500], "bandpass", fs=SR, output="sos"), rng.standard_normal(N)) * 0.012 * (env(0, 4.5, 1) + env(seg[9]["start"] - 0.5, T, 1))
music += rumble + wind
# duck under the voice
speech = np.convolve(np.abs(voice[:N]), np.ones(4800) / 4800, "same")
duck = 1 - 0.45 * np.clip(speech * 25, 0, 1)
out = voice[:N] * 1.0 + music * duck
out *= np.clip(t / 0.3, 0, 1) * np.clip((T - t) / 0.8, 0, 1)
out /= np.max(np.abs(out)) / 0.9
sf.write("/home/claude/work/final_audio.wav", out, SR)
print("ok", T)
