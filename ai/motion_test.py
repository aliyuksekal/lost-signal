"""Hareket testi: aynı karelerle (1) LTX 0.9.5 az hareketli komut + kısa klip, (2) Wan 2.2 TI2V 5B. Süre ve kalite karşılaştırması."""
import os, sys, json, time, traceback, urllib.request
OUT = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working"
REL = "https://github.com/aliyuksekal/lost-signal/releases/download/render-ai_poc_6/kare_%s.png"
SHOTS = {
    "emma": "a woman standing still in a dim bathroom, she slowly narrows her eyes and breathes, very subtle movement, static camera, photorealistic",
    "maggie": "an older woman holding a pie at a front door, she smiles and tilts her head slightly, very subtle movement, static camera, photorealistic",
    "daniel": "a man at a dinner table looks up from his phone with a calm face, very subtle movement, static camera, photorealistic",
}
NEG = "worst quality, fast motion, jittery, blurry, distorted face, deformed, morphing, camera shake"
REPORT = {}
def save():
    json.dump(REPORT, open(OUT + "/report.json", "w"), indent=1)

def get_imgs():
    from diffusers.utils import load_image
    imgs = {}
    for k in SHOTS:
        p = "/tmp/k_%s.png" % k
        urllib.request.urlretrieve(REL % k, p)
        imgs[k] = load_image(p)
    return imgs

def ltx(imgs):
    import torch
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import export_to_video
    pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    for k, prompt in SHOTS.items():
        t = time.time()
        fr = pipe(image=imgs[k].resize((768, 448)), prompt=prompt, negative_prompt=NEG, width=768, height=448, num_frames=65,
                  num_inference_steps=40, guidance_scale=3.5, generator=torch.Generator("cuda").manual_seed(5)).frames[0]
        export_to_video(fr, OUT + "/ltx_%s.mp4" % k, fps=24)
        REPORT["ltx_%s_sn" % k] = round(time.time() - t, 1); save()
    del pipe; torch.cuda.empty_cache()

def wan(imgs):
    import torch
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
    from diffusers.utils import export_to_video
    mid = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    for k, prompt in SHOTS.items():
        t = time.time()
        fr = pipe(image=imgs[k].resize((832, 480)), prompt=prompt, negative_prompt=NEG, width=832, height=480, num_frames=49,
                  num_inference_steps=30, guidance_scale=5.0, generator=torch.Generator("cuda").manual_seed(5)).frames[0]
        export_to_video(fr, OUT + "/wan_%s.mp4" % k, fps=24)
        REPORT["wan_%s_sn" % k] = round(time.time() - t, 1); save()
        if time.time() - t > 1800:
            REPORT["wan_not"] = "klip başına 30 dk'dan uzun, durduruldu"; break

if __name__ == "__main__":
    imgs = get_imgs()
    for name, fn in (("ltx", ltx), ("wan", wan)):
        try:
            fn(imgs)
        except Exception:
            REPORT[name + "_hata"] = traceback.format_exc()[-2500:]
        save()
