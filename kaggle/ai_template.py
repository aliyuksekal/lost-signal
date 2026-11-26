# Lost Signal - gerçekçi AI işi (Kaggle GPU). GitHub Actions __JOB__ alanını doldurup gönderir.
import os, json, subprocess, time, shutil

JOB = json.loads(r'''__JOB__''')
REPO = "https://github.com/aliyuksekal/lost-signal"
WORK = "/kaggle/working"
TMP = "/tmp/ls"
LOG = open(WORK + "/render_log.txt", "a")

def sh(cmd, env=None, check=True):
    t = time.time()
    print(">>", cmd, flush=True); LOG.write(">> %s\n" % cmd); LOG.flush()
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run(cmd, shell=True, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(r.stdout.splitlines()[-60:])
    print(tail, flush=True); LOG.write(tail + "\n(%.0f sn, kod %d)\n" % (time.time() - t, r.returncode)); LOG.flush()
    if check and r.returncode != 0:
        raise SystemExit("HATA: %s" % cmd)
    return r.returncode

json.dump(JOB, open(WORK + "/job.json", "w"))
shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP, exist_ok=True)
sh("git clone --depth 1 -b %s %s %s/repo" % (JOB.get("ref", "main"), REPO, TMP))
sh("nvidia-smi; df -h /tmp /kaggle/working; free -g", check=False)
sh("pip install -q -U " + " ".join(JOB.get("pip", [])), check=False)
for c in JOB.get("setup", []):
    sh(c, check=False)
sh("cd %s/repo/ai && python %s %s" % (TMP, JOB["script"], WORK), {"HF_HOME": "/tmp/hf", "LS_JOB": json.dumps(JOB)}, check=False)
LOG.write("BITTI\n"); LOG.close()
