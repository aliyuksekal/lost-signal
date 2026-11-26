"""Lost Signal - gerçekçi AI deneme (Kaggle T4).
Adımlar: (1) SDXL RealVis ile gerçekçi karakter kareleri, (2) LTX-Video ile kareden video,
(3) Kokoro ile doğal İngilizce ses, (4) birleştirme. Her adımın süresi ve hatası report.json'a yazılır.
"""
import os, sys, json, time, subprocess, traceback, glob, shutil
OUT = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working"
TMP = "/tmp/ai"; os.makedirs(TMP, exist_ok=True)
REPORT = {"steps": {}}

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

STYLE = ("cinematic film still, photorealistic, 35mm film grain, shallow depth of field, moody night lighting, "
         "warm desk lamp and cold blue moonlight, highly detailed skin texture, natural pose")
NEG = "cartoon, anime, 3d render, cgi, painting, illustration, deformed face, extra fingers, bad hands, text, watermark, logo, blurry"
SHOTS = [
    dict(id="nora", seed=11,
         img="a 23-year-old woman with a shoulder-length copper red bob haircut and freckles, green eyes, mustard yellow raincoat, "
             "one eyebrow raised in a skeptical look, standing in a small dark bedroom, an old wooden vintage radio with a glowing amber dial on the desk beside her",
         mov="The woman slowly raises one eyebrow and turns her head toward the glowing radio, subtle natural movement, the lamp light flickers softly, static camera",
         voice="af_heart", line="Radios don't talk, Benny. Someone is pranking us."),
    dict(id="kai", seed=22,
         img="a 23-year-old East Asian man with messy black hair and round wire glasses, navy hoodie, leaning over an old wooden vintage radio "
             "with a glowing amber dial on a cluttered desk, focused curious expression, dark bedroom, window with moonlight behind him",
         mov="The young man leans closer to the radio and slowly turns the tuning knob, the amber dial glows brighter, subtle camera push in",
         voice="am_michael", line="Then how does it know our names?"),
    dict(id="benny", seed=33,
         img="a 22-year-old Black man with curly hair under a backwards red baseball cap, orange and white striped t-shirt, headphones around his neck, "
             "wide eyes and open mouth in shock, dark bedroom at night, an old wooden vintage radio glowing on the desk in front of him",
         mov="The young man jumps back slightly in shock, eyes widening, then leans in toward the radio, handheld camera feel",
         voice="am_fenrir", line="Whoa. Did that radio just... talk?"),
]
RADIO_LINE = ("bm_george", "Nora. Benny. Kai. Tomorrow, three o'clock. Do not cross the bridge.")

@step("1_karakter_kareleri")
def keyframes():
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    pipe = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V5.0", torch_dtype=torch.float16, variant="fp16",
                                                     use_safetensors=True).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.enable_vae_tiling()
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
    for s in SHOTS:
        src = OUT + "/kare_%s.png" % s["id"]
        if not os.path.exists(src):
            continue
        small = TMP + "/in_%s.png" % s["id"]
        Image.open(src).resize((768, 448)).save(small)
        outdir = TMP + "/ltx_%s" % s["id"]
        t = time.time()
        sh("cd %s/LTX-Video && python inference.py --prompt %s --conditioning_media_paths %s --conditioning_start_frames 0 "
           "--height 448 --width 768 --num_frames 97 --frame_rate 24 --seed %d --pipeline_config %s --output_path %s"
           % (TMP, json.dumps(s["mov"] + ", photorealistic, cinematic"), small, s["seed"], cfg, outdir))
        vids = sorted(glob.glob(outdir + "/**/*.mp4", recursive=True))
        if vids:
            shutil.copy(vids[-1], OUT + "/klip_%s.mp4" % s["id"])
        times[s["id"]] = round(time.time() - t, 1)
    return times

@step("3_ses")
def voices():
    import soundfile as sf, numpy as np
    from kokoro import KPipeline
    res = {}
    for voice, text, name in [(s["voice"], s["line"], s["id"]) for s in SHOTS] + [(RADIO_LINE[0], RADIO_LINE[1], "radio")]:
        lang = "b" if voice.startswith("b") else "a"
        pipe = KPipeline(lang_code=lang)
        audio = np.concatenate([a for _, _, a in pipe(text, voice=voice)])
        sf.write(OUT + "/ses_%s.wav" % name, audio, 24000)
        res[name] = round(len(audio) / 24000, 2)
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
    sh("ffmpeg -y -v error -f concat -safe 0 -i %s/list.txt -c copy %s/LostSignal_gercekci_test.mp4" % (TMP, OUT))
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
