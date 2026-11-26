"""Lost Signal teaser ses miksajı: diyaloglar, radyo efekti, parazit, ortam, müzik."""
import json, wave
import numpy as np
from scipy.signal import butter, sosfilt, resample_poly

SR = 48000
DUR = 30.0
import os
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio") + "/"
OUTWAV = os.environ.get("LS_OUT", os.path.dirname(os.path.abspath(__file__))) + "/mix.wav"
N = int(SR * DUR)
rng = np.random.default_rng(1)
t = np.arange(N) / SR

def load(name):
    w = wave.open(BASE + name + ".wav")
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return resample_poly(a, SR, w.getframerate()).astype(np.float32)

def bp(x, lo, hi, order=4):
    return sosfilt(butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x)

def lp(x, hi):
    return sosfilt(butter(2, hi, "lowpass", fs=SR, output="sos"), x)

def place(buf, x, start, gain=1.0):
    i = int(start * SR); j = min(N, i + len(x)); buf[i:j] += x[: j - i] * gain

def env(a, b, attack=0.5, release=0.5):
    e = np.zeros(N)
    i, j = int(a * SR), int(b * SR)
    e[i:j] = 1
    ka = int(attack * SR); kr = int(release * SR)
    e[i:i + ka] = np.linspace(0, 1, ka)
    e[j - kr:j] = np.linspace(1, 0, kr)
    return e

voice = np.zeros(N); radio = np.zeros(N); fx = np.zeros(N); music = np.zeros(N)

for ln in json.load(open(BASE + "timeline.json")):
    x = load(ln["file"])
    if ln["who"] == "radio":
        y = bp(x, 350, 3200)
        y = np.tanh(y * 3.5) / 2.2
        trem = 1 - 0.25 * (0.5 + 0.5 * np.sin(2 * np.pi * 5.5 * np.arange(len(y)) / SR))
        drop = np.ones(len(y))  # sinyal kopmaları
        for k in range(6):
            c = rng.integers(0, len(y) - 4000); drop[c:c + rng.integers(1500, 4000)] = 0.15
        place(radio, y * trem * drop, ln["start"], 0.9)
    else:
        place(voice, x, ln["start"], 0.85)

# radyo paraziti (statik)
white = rng.standard_normal(N)
static = bp(white, 800, 7000) * 0.06
crackle = (rng.random(N) > 0.9993) * rng.standard_normal(N) * 0.6
crackle = lp(crackle, 5000)
st_env = (np.clip((t - 0.3) / 3.5, 0, 1) * (t < 4.0) * 0.9
          + env(4.0, 7.8, 0.1, 0.3) * 0.6 + env(18.6, 24.4, 0.2, 0.4) * 0.6
          + env(7.8, 18.6, 0.3, 0.3) * 0.12 + env(24.4, 27.4, 0.2, 0.3) * 0.25)
# frekans tarama "vuu" sesi (radyo ayarlanırken)
sweep_f = 400 + 1800 * np.abs(np.sin(2 * np.pi * 0.35 * t))
sweep = np.sin(2 * np.pi * np.cumsum(sweep_f) / SR) * 0.03 * (t < 4.0) * np.clip((t - 1.0) / 2, 0, 1)
fx += (static + crackle) * st_env + sweep

# oda tonu
room = lp(rng.standard_normal(N), 300) * 0.012
fx += room

# müzik: karanlık drone + nabız
def tone(freq, amp):
    return amp * (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2.003 * t))
pad = tone(55, 0.05) + tone(82.4, 0.035) + tone(110.3, 0.025) + tone(130.8, 0.018)  # A1, E2, A2, C3 (minör his)
pad *= np.clip(t / 6, 0, 1) * (1 + 0.6 * np.clip((t - 18.8) / 5, 0, 1)) * (t < 27.5)
pulse = np.zeros(N)
for b in np.arange(18.9, 27.4, 0.75):  # kalp atışı gibi nabız
    i = int(b * SR); k = int(0.35 * SR)
    pulse[i:i + k] += np.sin(2 * np.pi * 50 * np.arange(k) / SR) * np.exp(-np.arange(k) / (0.08 * SR)) * 0.25
music += pad + pulse
# final darbe (başlık kartı)
i = int(27.5 * SR); k = int(2.5 * SR); tt = np.arange(k) / SR
hit = np.sin(2 * np.pi * (60 - 25 * tt) * tt) * np.exp(-tt / 0.6) * 0.6
hit += bp(rng.standard_normal(k), 200, 4000) * np.exp(-tt / 0.15) * 0.25
music[i:i + k] += hit
# son 2.5 sn'de hafif statik kuyruğu
fx[i:i + k] += bp(rng.standard_normal(k), 1000, 6000) * 0.03 * np.exp(-tt / 1.0)

# müzik konuşmada biraz kısılsın (ducking)
speech = np.abs(voice) + np.abs(radio)
duck = 1 - 0.45 * np.clip(lp(speech, 5) * 12, 0, 1)
mix = voice + radio + fx + music * duck
mix /= np.max(np.abs(mix)) / 0.89
fade = np.ones(N); fade[-int(0.8 * SR):] = np.linspace(1, 0, int(0.8 * SR))
mix *= fade
out = wave.open(OUTWAV, "w"); out.setnchannels(1); out.setsampwidth(2); out.setframerate(SR)
out.writeframes((mix * 32767).astype(np.int16).tobytes()); out.close()
print("mix ok")
