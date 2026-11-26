"""Lost Signal - gerçekçi AI deneme (Kaggle T4).
Adımlar: (1) SDXL RealVis ile gerçekçi karakter kareleri, (2) LTX-Video ile kareden video,
(3) Kokoro ile doğal İngilizce ses, (4) birleştirme. Her adımın süresi ve hatası report.json'a yazılır.
"""
import os, sys, json, time, subprocess, traceback, glob, shutil
OUT = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working"
TMP = "/tmp/ai"; os.makedirs(TMP, exist_ok=True)
REPORT = {"steps": {}}
HELP = ""

def step(name):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            try:
                r = fn(*a, **k)
                REPORT["steps"][name] = {"ok": True, "sn": round(time.time() - t, 1), "info": r}
                return r
            except Exception as e:
                REPORT["steps"][name] = {"ok": False, "sn": round(time.time() - t, 1), "hata": repr(e)[:800],
                                         "iz": traceback.format_exc()[-2500:]}
                print("HATA", name, e, flush=True)
                return None
            finally:
                json.dump(REPORT, open(OUT + "/report.json", "w"), indent=1)
        return run
    return deco

def sh(cmd):
    print(">>", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        raise RuntimeError("komut hata verdi: %s\n%s" % (cmd, r.stdout[-1500:]))
    return r.stdout

STYLE = ("cinematic film still from a prestige thriller TV series, photorealistic, 35mm film grain, shallow depth of field, "
         "moody lighting, highly detailed natural skin texture, natural pose")
NEG = "cartoon, anime, 3d render, cgi, painting, illustration, deformed face, extra fingers, bad hands, text, watermark, logo, blurry, oversaturated"
SHOTS = [
    dict(id="emma", seed=11,
         img="32-year-old woman, shoulder-length dark brown wavy hair, hazel eyes, light olive skin, minimal makeup, cream knit sweater, "
             "standing in a dim bathroom at night, staring with a worried face at a large mirror, a tiny red light glowing behind the glass",
         mov="The woman slowly leans toward the mirror, her eyes narrowing as she notices the small red light, tense subtle movement, slow camera push in",
         voice="af_heart", line="Daniel... Daniel, wake up. There's a camera behind the mirror."),
    dict(id="maggie", seed=22,
         img="64-year-old woman, silver hair in a neat bob, pearl earrings, pastel cardigan, holding a homemade cherry pie, "
             "standing at the front door of a perfect suburban house on a sunny morning, wide perfect smile that does not reach her eyes",
         mov="The older woman tilts her head and smiles wider, her eyes stay cold, she lifts the pie slightly, static camera",
         voice="af_bella", line="Welcome to Willow Lane! We're a family here. We look out for each other."),
    dict(id="daniel", seed=33,
         img="35-year-old man, short neat light brown hair, light stubble, blue eyes, fair skin, navy button-up shirt with rolled sleeves, "
             "sitting at a dinner table between moving boxes at night, looking down at his glowing phone with a guilty expression",
         mov="The man glances at his phone, quickly turns the screen face down and looks up with a forced calm smile, slow camera push in",
         voice="am_michael", line="It's just work. Don't worry about it."),
]
RADIO_LINE = ("bf_emma", "If you're watching this... don't trust them.")

@step("1_karakter_kareleri")
def keyframes():
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    pipe = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V5.0", torch_dtype=torch.float16, variant="fp16",
                                                     use_safetensors=True).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    times = {}
    for s in SHOTS:
        t = time.time()
        g = torch.Generator("cuda").manual_seed(s["seed"])
        im = pipe(prompt=s["img"] + ", " + STYLE, negative_prompt=NEG, width=1216, height=704,
                  num_inference_steps=28, guidance_scale=5.0, generator=g).images[0]
        im.save(OUT + "/kare_%s.png" % s["id"])
        times[s["id"]] = round(time.time() - t, 1)
    del pipe; torch.cuda.empty_cache()
    return times

@step("2a_ltx_kurulum")
def ltx_setup():
    sh("cd %s && git clone --depth 1 https://github.com/Lightricks/LTX-Video.git" % TMP)
    sh("cd %s/LTX-Video && pip install -q -e .[inference]" % TMP)
    return sorted(os.path.basename(p) for p in glob.glob(TMP + "/LTX-Video/configs/*.yaml"))

@step("2b_kareden_video")
def img2vid(cfg_name):
    from PIL import Image
    times = {}
    cfg = "%s/LTX-Video/configs/%s" % (TMP, cfg_name)
    global HELP
    try:
        HELP = sh("cd %s/LTX-Video && python inference.py --help" % TMP)
    except Exception as e:
        HELP = str(e)
    REPORT["ltx_offload"] = "offload_to_cpu" in HELP
    for s in SHOTS:
        src = OUT + "/kare_%s.png" % s["id"]
        if not os.path.exists(src):
            continue
        small = TMP + "/in_%s.png" % s["id"]
        Image.open(src).resize((768, 448)).save(small)
        outdir = TMP + "/ltx_%s" % s["id"]
        t = time.time()
        extra = "--offload_to_cpu" if "offload_to_cpu" in HELP else ""
        sh("cd %s/LTX-Video && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inference.py " + extra + " --prompt %s --conditioning_media_paths %s --conditioning_start_frames 0 "
           "--height 448 --width 768 --num_frames 97 --frame_rate 24 --seed %d --pipeline_config %s --output_path %s"
           % (TMP, json.dumps(s["mov"] + ", photorealistic, cinematic"), small, s["seed"], cfg, outdir))
        vids = sorted(glob.glob(outdir + "/**/*.mp4", recursive=True))
        if vids:
            shutil.copy(vids[-1], OUT + "/klip_%s.mp4" % s["id"])
        times[s["id"]] = round(time.time() - t, 1)
    return times

@step("3_ses")
def voices():
    import soundfile as sf, urllib.request
    from kokoro_onnx import Kokoro
    base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        if not os.path.exists(TMP + "/" + f):
            urllib.request.urlretrieve(base + f, TMP + "/" + f)
    k = Kokoro(TMP + "/kokoro-v1.0.onnx", TMP + "/voices-v1.0.bin")
    res = {}
    for voice, text, name in [(s["voice"], s["line"], s["id"]) for s in SHOTS] + [(RADIO_LINE[0], RADIO_LINE[1], "claire")]:
        audio, sr = k.create(text, voice=voice, speed=1.0, lang="en-gb" if voice.startswith("b") else "en-us")
        sf.write(OUT + "/ses_%s.wav" % name, audio, sr)
        res[name] = round(len(audio) / sr, 2)
    return res

@step("4_birlestir")
def assemble():
    parts = []
    for s in SHOTS:
        v, a = OUT + "/klip_%s.mp4" % s["id"], OUT + "/ses_%s.wav" % s["id"]
        if not os.path.exists(v):
            continue
        o = TMP + "/p_%s.mp4" % s["id"]
        if os.path.exists(a):
            sh("ffmpeg -y -v error -i %s -i %s -filter_complex \"[1:a]apad[a]\" -map 0:v -map \"[a]\" -shortest "
               "-vf scale=1280:-2 -c:v libx264 -crf 18 -c:a aac %s" % (v, a, o))
        else:
            sh("ffmpeg -y -v error -i %s -f lavfi -i anullsrc=r=24000:cl=mono -shortest -vf scale=1280:-2 -c:v libx264 -crf 18 -c:a aac %s" % (v, o))
        parts.append(o)
    if not parts:
        return "klip yok"
    open(TMP + "/list.txt", "w").write("".join("file '%s'\n" % p for p in parts))
    sh("ffmpeg -y -v error -f concat -safe 0 -i %s/list.txt -c copy %s/PerfectNeighbors_gercekci_test.mp4" % (TMP, OUT))
    return len(parts)

if __name__ == "__main__":
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    keyframes()
    cfgs = ltx_setup() or []
    pick = [c for c in cfgs if "2b" in c and "distilled" in c]
    pick = sorted(pick)[-1] if pick else None
    REPORT["ltx_config"] = pick
    if pick:
        img2vid(pick)
    voices()
    assemble()
    json.dump(REPORT, open(OUT + "/report.json", "w"), indent=1)
    print(json.dumps(REPORT, indent=1)[:3000])
