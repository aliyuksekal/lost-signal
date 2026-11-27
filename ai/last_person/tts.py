import json, numpy as np, soundfile as sf, sys
from kokoro_onnx import Kokoro
spec = json.load(open("/home/claude/lost-signal/ai/last_person/spec.json"))
k = Kokoro("/tmp/kk/kokoro-v1.0.onnx", "/tmp/kk/voices-v1.0.bin")
voice = sys.argv[1] if len(sys.argv) > 1 else spec["voice"]; speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SR = 24000; parts = []; t = 0.35; segs = []
gaps = [0.55,0.5,0.6,0.5,0.55,0.75,0.6,0.7,0.6,0.5]
for i, s in enumerate(spec["segments"]):
    a, sr = k.create(s["text"], voice=voice, speed=speed, lang="en-gb")
    # trim silence
    idx = np.where(np.abs(a) > 0.01)[0]; a = a[max(0, idx[0]-600): idx[-1]+1200]
    d = len(a)/SR; segs.append({"start": round(t,3), "end": round(t+d,3)}); parts.append((t, a)); t += d + gaps[i]
total = t + 0.6
out = np.zeros(int(total*SR))
for st, a in parts: i = int(st*SR); out[i:i+len(a)] += a
sf.write("/home/claude/work/narration_%s.wav" % voice, out, SR)
print(voice, round(total,2), [round(s["end"]-s["start"],2) for s in segs])
json.dump({"total": round(total,3), "segments": segs}, open("/home/claude/work/timing_%s.json" % voice, "w"))
