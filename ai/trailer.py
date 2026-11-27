"""The Perfect Neighbors - dikey fragman (Shorts), ~40 sn.
Karakter modelleri (LoRA) ile dikey ana kareler -> LTX ile az hareketli klipler -> anlatıcı sesi + müzik -> yazı kartları.
"""
import os, sys, json, time, traceback, subprocess, urllib.request
import numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working"
TMP = "/tmp/tr"; os.makedirs(TMP, exist_ok=True)
REL = "https://github.com/aliyuksekal/lost-signal/releases/download/render-karakter_egitimi_1/tpn_%s.safetensors"
BASE = "SG161222/RealVisXL_V5.0"
W, H = 512, 896          # LTX boyutu (dikey)
KW, KH = 768, 1344       # ana kare boyutu
FPS = 24; NF = 65        # ~2.7 sn klip
REPORT = {"kareler": {}, "klipler": {}}
def save(): json.dump(REPORT, open(OUT + "/report.json", "w"), indent=1)

LOOK = {
    "emma": "woman, dark brown wavy shoulder-length hair, cream knit sweater",
    "daniel": "man, short light brown hair, light stubble, navy button-up shirt",
    "maggie": "woman, silver bob haircut, pearl earrings, pastel cardigan",
    "vanessa": "woman, long sleek platinum blonde hair, white silk blouse",
    "owen": "man, messy dark hair, thick black-rimmed glasses, grey hoodie",
    "rosa": "woman, grey-streaked black hair in a low bun, denim shirt",
    "claire": "woman, short platinum pixie cut, black turtleneck",
}
STYLE = "vertical cinematic film still from a prestige psychological thriller TV series, photorealistic, 35mm film grain, shallow depth of field, moody color grade"
NEG = "cartoon, anime, 3d render, deformed face, extra fingers, text, watermark, logo, blurry, oversaturated"
# (id, karakter, ana kare tarifi, hareket tarifi)
SHOTS = [
    ("s01", None, "a perfect quiet suburban cul-de-sac street at golden hour, identical white houses with green lawns and white picket fences, a street sign that reads nothing",
     "slow cinematic push in along the perfect suburban street, leaves gently moving, static calm"),
    ("s02", "maggie", "standing at the front door of a white suburban house holding a homemade cherry pie, wide polite smile, cold eyes, sunny morning",
     "she tilts her head slightly and her smile widens, very subtle movement, static camera"),
    ("s03", "vanessa", "standing by a glowing backyard swimming pool at dusk holding a glass of white wine, perfect smile, sad eyes",
     "she slowly sips her wine and looks away, very subtle movement, static camera"),
    ("s04", "owen", "peeking through the slats of window blinds from a dark room, watching the street, face half in shadow",
     "he slowly parts the blinds with two fingers and watches, very subtle movement, static camera"),
    ("s05", "emma", "standing among cardboard moving boxes in an empty living room, looking up at the house with unease, daylight",
     "she slowly turns her head and looks around the room, very subtle movement, slow push in"),
    ("s06", "daniel", "sitting alone at a kitchen table at night lit by the glow of his phone, guilty tense expression",
     "he glances at his phone and quickly turns it face down, subtle movement, static camera"),
    ("s07", None, "close-up of a bathroom mirror at night, a tiny red recording light glowing behind the glass, dark moody bathroom",
     "the small red light blinks slowly behind the mirror glass, very slow push in, static"),
    ("s08", "rosa", "sitting on a dark front porch at night holding binoculars, watching the house across the street, porch light",
     "she slowly lowers the binoculars, eyes narrowing, very subtle movement, static camera"),
    ("s09", "claire", "in a dark room lit by a single cold light, looking straight into the camera, frightened, whispering",
     "she leans slightly toward the camera, eyes wide, very subtle movement, static camera"),
    ("s10", "emma", "in a dark bathroom at night, close-up, staring at a mirror in fear, red light reflected in her eyes",
     "her eyes widen in fear and she holds her breath, very subtle movement, slow push in"),
]
VO = [  # (zaman sn, ses, metin)
    (0.4, "am_onyx", "Willow Lane. Perfect houses."),
    (3.0, "af_bella", "Welcome to Willow Lane!"),
    (5.6, "am_onyx", "Perfect smiles."),
    (8.4, "am_onyx", "Perfect neighbors."),
    (11.2, "am_onyx", "But the family who lived here before..."),
    (14.2, "am_onyx", "vanished overnight."),
    (17.0, "am_onyx", "And someone... is still watching."),
    (22.6, "bf_emma", "If you're watching this... don't trust them."),
]
CARDS = [(27.4, 3.4, ["THE PERFECT", "NEIGHBORS"], "Everyone on Willow Lane has a secret."),
         (30.8, 2.6, ["EPISODE 1"], "Coming soon")]

def sh(c):
    r = subprocess.run(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode: raise RuntimeError(r.stdout[-1500:])
    return r.stdout

def keyframes():
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    pipe = StableDiffusionXLPipeline.from_pretrained(BASE, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    loaded = None
    for sid, ch, desc, _ in SHOTS:
        t = time.time()
        try:
            if ch != loaded:
                if loaded: pipe.unload_lora_weights()
                if ch:
                    p = TMP + "/tpn_%s.safetensors" % ch
                    if not os.path.exists(p): urllib.request.urlretrieve(REL % ch, p)
                    pipe.load_lora_weights(p)
                loaded = ch
            who = ("photo of tpn_%s %s, " % (ch, LOOK[ch])) if ch else ""
            im = pipe(prompt=who + desc + ", " + STYLE, negative_prompt=NEG, width=KW, height=KH, num_inference_steps=30,
                      guidance_scale=5.0, generator=torch.Generator("cuda").manual_seed(int(sid[1:]) * 17)).images[0]
            im.save(OUT + "/kare_%s.png" % sid)
            REPORT["kareler"][sid] = round(time.time() - t, 1)
        except Exception:
            REPORT["kareler"][sid] = "HATA " + traceback.format_exc()[-600:]
        save()
    del pipe; torch.cuda.empty_cache()

def clips():
    import torch
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image
    pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    for sid, ch, _, mov in SHOTS:
        src = OUT + "/kare_%s.png" % sid
        if not os.path.exists(src): continue
        t = time.time()
        try:
            fr = pipe(image=load_image(src).resize((W, H)), prompt=mov + ", photorealistic, cinematic",
                      negative_prompt="worst quality, fast motion, jittery, blurry, distorted face, deformed, morphing",
                      width=W, height=H, num_frames=NF, num_inference_steps=40, guidance_scale=3.5,
                      generator=torch.Generator("cuda").manual_seed(5)).frames[0]
            export_to_video(fr, TMP + "/klip_%s.mp4" % sid, fps=FPS)
            REPORT["klipler"][sid] = round(time.time() - t, 1)
        except Exception:
            REPORT["klipler"][sid] = "HATA " + traceback.format_exc()[-600:]
        save()

def audio(total):
    import soundfile as sf
    from kokoro_onnx import Kokoro
    from scipy.signal import butter, sosfilt
    base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        if not os.path.exists(TMP + "/" + f): urllib.request.urlretrieve(base + f, TMP + "/" + f)
    k = Kokoro(TMP + "/kokoro-v1.0.onnx", TMP + "/voices-v1.0.bin")
    SR = 24000; N = int(total * SR); t = np.arange(N) / SR
    mix = np.zeros(N); rng = np.random.default_rng(3)
    for at, voice, text in VO:
        a, sr = k.create(text, voice=voice, speed=0.92 if voice == "am_onyx" else 1.0, lang="en-gb" if voice.startswith("b") else "en-us")
        if voice == "bf_emma":  # Claire: fısıltı + kayıt cihazı hissi
            a = sosfilt(butter(4, [300, 3500], "bandpass", fs=sr, output="sos"), a) * 1.3
        i = int(at * SR); j = min(N, i + len(a)); mix[i:j] += a[: j - i] * 0.9
    def tone(f, amp): return amp * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.003 * t))
    pad = (tone(55, .05) + tone(82.4, .03) + tone(110, .02) + tone(130.8, .015)) * np.clip(t / 4, 0, 1) * (1 + np.clip((t - 16) / 8, 0, 1))
    for b in np.arange(16.5, 27.0, 0.6):  # kalp atışı
        i = int(b * SR); n = int(.3 * SR); pad[i:i + n] += np.sin(2 * np.pi * 48 * np.arange(n) / SR) * np.exp(-np.arange(n) / (.07 * SR)) * .3
    for hit in (27.4, 30.8):  # vuruşlar
        i = int(hit * SR); n = min(int(2.5 * SR), N - i); tt = np.arange(n) / SR
        pad[i:i + n] += np.sin(2 * np.pi * (60 - 25 * tt) * tt) * np.exp(-tt / .6) * .6
    hiss = sosfilt(butter(2, [1500, 7000], "bandpass", fs=SR, output="sos"), rng.standard_normal(N)) * .015 * ((t > 22) & (t < 27))
    speech = np.abs(mix); duck = 1 - .4 * np.clip(np.convolve(speech, np.ones(2400) / 2400, "same") * 15, 0, 1)
    out = mix + (pad + hiss) * duck
    out /= np.max(np.abs(out)) / .89
    sf.write(TMP + "/trailer.wav", out, SR)

def card(lines, sub, w=1080, h=1920):
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import matplotlib
    fb = os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSerif-Bold.ttf")
    fr = os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans.ttf")
    im = Image.new("RGB", (w, h), (6, 6, 8)); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(fb, 118); f2 = ImageFont.truetype(fr, 44)
    y = h // 2 - 70 * len(lines)
    glow = Image.new("RGB", (w, h)); gd = ImageDraw.Draw(glow)
    for i, ln in enumerate(lines):
        gd.text(((w - d.textlength(ln, font=f)) / 2, y + i * 140), ln, font=f, fill=(170, 20, 20))
    glow = glow.filter(ImageFilter.GaussianBlur(18)); im = Image.composite(glow, im, glow.convert("L")); d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        d.text(((w - d.textlength(ln, font=f)) / 2, y + i * 140), ln, font=f, fill=(238, 232, 222))
    d.text(((w - d.textlength(sub, font=f2)) / 2, y + len(lines) * 140 + 40), sub, font=f2, fill=(200, 196, 190))
    return im

def assemble():
    import imageio_ffmpeg, shutil
    ff = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    parts = []
    for sid, *_ in SHOTS:
        p = TMP + "/klip_%s.mp4" % sid
        if os.path.exists(p):
            q = TMP + "/v_%s.mp4" % sid
            sh('%s -y -v error -i %s -vf "scale=1080:1890:flags=lanczos,pad=1080:1920:0:15:black,eq=contrast=1.06:saturation=0.9,fps=24,format=yuv420p" -an -c:v libx264 -crf 18 %s' % (ff, p, q))
            parts.append(q)
    for at, dur, lines, sub in CARDS:
        im = card(lines, sub); ip = TMP + "/card_%d.png" % int(at); im.save(ip); q = TMP + "/c_%d.mp4" % int(at)
        sh('%s -y -v error -loop 1 -t %.2f -i %s -vf "fade=in:st=0:d=0.4,fps=24,format=yuv420p" -c:v libx264 -crf 18 %s' % (ff, dur, ip, q))
        parts.append(q)
    open(TMP + "/list.txt", "w").write("".join("file '%s'\n" % p for p in parts))
    sh("%s -y -v error -f concat -safe 0 -i %s/list.txt -c copy %s/video.mp4" % (ff, TMP, TMP))
    total = len([p for p in parts if "/v_" in p]) * NF / FPS + sum(c[1] for c in CARDS)
    audio(total)
    sh("%s -y -v error -i %s/video.mp4 -i %s/trailer.wav -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart %s/ThePerfectNeighbors_Fragman_Shorts.mp4" % (ff, TMP, TMP, OUT))

if __name__ == "__main__":
    for name, fn in (("kareler", keyframes), ("klipler", clips), ("birlestir", assemble)):
        t = time.time()
        try:
            fn(); REPORT[name + "_sn"] = round(time.time() - t)
        except Exception:
            REPORT[name + "_hata"] = traceback.format_exc()[-2500:]
        save()
