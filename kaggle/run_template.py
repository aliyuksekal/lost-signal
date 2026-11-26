# Lost Signal - Kaggle GPU render işi.
# GitHub Actions bu dosyayı __JOB__ yerine iş bilgisini koyup Kaggle'a gönderir.
import os, json, subprocess, shutil, glob, time

JOB = json.loads(r'''__JOB__''')
REPO = "https://github.com/aliyuksekal/lost-signal"
BLENDER_URL = "https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz"
WORK = "/kaggle/working"
TMP = "/tmp/ls"
OUT = TMP + "/out"
LOG = open(WORK + "/render_log.txt", "a")

def sh(cmd, env=None):
    t = time.time()
    print(">>", cmd, flush=True); LOG.write(">> %s\n" % cmd); LOG.flush()
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run(cmd, shell=True, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(r.stdout.splitlines()[-40:])
    print(tail, flush=True); LOG.write(tail + "\n(%.0f sn)\n" % (time.time() - t)); LOG.flush()
    if r.returncode != 0:
        raise SystemExit("HATA: %s" % cmd)

json.dump(JOB, open(WORK + "/job.json", "w"))
shutil.rmtree(TMP, ignore_errors=True); os.makedirs(OUT, exist_ok=True)
sh("git clone --depth 1 -b %s %s %s/repo" % (JOB.get("ref", "main"), REPO, TMP))
sh("cd %s && curl -sSL -o blender.tar.xz %s && tar -xf blender.tar.xz && mv blender-*-linux-x64 blender" % (TMP, BLENDER_URL))
sh("pip install -q scipy matplotlib imageio-ffmpeg")
sh("nvidia-smi || true")

S = TMP + "/repo/studio"
B = TMP + "/blender/blender"
env = {"LS_OUT": OUT, "LS_QUALITY": JOB.get("quality", "hq"), "LS_GPU": "1",
       "VERT": "1" if JOB.get("orientation") == "v" else "0"}
sh("cd %s && python mix.py" % S, env)
sh("%s -b -P %s/scene.py -- build" % (B, S), env)
sh("%s -b -P %s/scene.py -- render 1 %d" % (B, S, JOB.get("end_frame", 660)), env)
frames = OUT + ("/frames_v" if env["VERT"] == "1" else "/frames")
name = "LostSignal_%s.mp4" % JOB["name"]
sh("cd %s && python finish.py %s %s/mix.wav subs.srt %s/%s" % (S, frames, OUT, WORK, name), env)
print("BITTI:", name)
LOG.write("BITTI %s\n" % name); LOG.close()
