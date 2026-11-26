"""The Perfect Neighbors - karakter modelleri (LoRA) eğitimi, Kaggle 2x T4.
LS_JOB içinde: {"cast": {"emma": "<seçilen yüz görselinin URL'si>", ...}, "steps": 800}
Her karakter için: (1) seçilen yüzden IP-Adapter ile 24 çeşitli görsel, (2) LoRA eğitimi, (3) test görselleri.
İki ekran kartı paralel kullanılır: karakterler GPU 0 ve GPU 1'e dağıtılır.
"""
import os, sys, json, time, subprocess, traceback, urllib.request
OUT = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working"
JOB = json.loads(os.environ.get("LS_JOB", "{}"))
TMP = "/tmp/cast"
BASE = "SG161222/RealVisXL_V5.0"

DESC = {
    "emma": ("woman", "dark brown wavy shoulder-length hair, cream knit sweater"),
    "daniel": ("man", "short light brown hair, light stubble, navy button-up shirt"),
    "maggie": ("woman", "silver bob haircut, pearl earrings, pastel cardigan"),
    "grant": ("man", "salt-and-pepper swept back hair, grey polo shirt"),
    "vanessa": ("woman", "long sleek platinum blonde hair, white silk blouse"),
    "owen": ("man", "messy dark hair, thick black-rimmed glasses, grey hoodie"),
    "rosa": ("woman", "grey-streaked black hair in a low bun, denim shirt"),
    "claire": ("woman", "short platinum pixie cut, black turtleneck"),
}
VARIANTS = [
    "close-up portrait, looking at camera, neutral expression, soft window light",
    "close-up portrait, three-quarter view facing left, worried expression, warm lamp light at night",
    "close-up portrait, three-quarter view facing right, slight smile, daylight",
    "side profile, looking out of a window, cold blue moonlight",
    "medium shot, standing in a suburban kitchen, talking, natural light",
    "medium shot, sitting on a sofa at night, tense expression, warm lamp light",
    "medium shot, outdoors on a sunny suburban street, friendly expression",
    "close-up, frightened wide eyes, dark room, rim light",
    "close-up, angry frown, overcast light",
    "close-up, laughing, golden hour light",
    "medium shot, holding a phone, looking down, screen glow on the face",
    "close-up, crying softly, night, single light source",
]
SEEDS = [1, 2]  # her varyant için 2 tohum -> 24 görsel

def log(*a):
    print(*a, flush=True)

def build_dataset(name, gpu):
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import load_image
    sex, look = DESC[name]
    ref = load_image(TMP + "/ref_%s.png" % name)
    pipe = StableDiffusionXLPipeline.from_pretrained(BASE, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda:%d" % gpu)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
                         image_encoder_folder="models/image_encoder")
    pipe.set_ip_adapter_scale(0.75)
    d = TMP + "/ds_%s" % name; os.makedirs(d, exist_ok=True)
    ref.save(d + "/000.png")
    open(d + "/000.txt", "w").write("photo of tpn_%s %s, %s" % (name, sex, look))
    n = 1
    for v in VARIANTS:
        for sd in SEEDS:
            g = torch.Generator("cuda:%d" % gpu).manual_seed(sd * 1000 + n)
            prompt = "photorealistic cinematic photo of a %s, %s, %s, 35mm film, detailed skin" % (sex, look, v)
            im = pipe(prompt=prompt, ip_adapter_image=ref, negative_prompt="cartoon, 3d, deformed, text, watermark, blurry",
                      width=896, height=1152 if "close" in v else 896, num_inference_steps=26, guidance_scale=5.0, generator=g).images[0]
            im.save(d + "/%03d.png" % n)
            open(d + "/%03d.txt" % n, "w").write("photo of tpn_%s %s, %s, %s" % (name, sex, look, v))
            n += 1
    del pipe; torch.cuda.empty_cache()
    return d

def worker(names, gpu):
    """Bir ekran kartında sırayla: veri seti + eğitim + test."""
    rep = {}
    for name in names:
        r = rep[name] = {}
        try:
            t = time.time(); d = build_dataset(name, gpu); r["veri_seti_sn"] = round(time.time() - t)
            t = time.time()
            out = TMP + "/lora_%s" % name
            cmd = ("CUDA_VISIBLE_DEVICES=%d accelerate launch --num_processes 1 --mixed_precision fp16 %s/train_lora.py "
                   "--pretrained_model_name_or_path %s --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix "
                   "--dataset_name %s --caption_column text --image_column image "
                   "--instance_prompt 'photo of tpn_%s' --resolution 768 --train_batch_size 1 --gradient_accumulation_steps 2 "
                   "--gradient_checkpointing --use_8bit_adam --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 "
                   "--max_train_steps %d --rank 16 --mixed_precision fp16 --seed 0 --output_dir %s "
                   % (gpu, TMP, BASE, d + "_hf", name, JOB.get("steps", 800), out))
            to_hf_folder(d)
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            open(OUT + "/egitim_log_%s.txt" % name, "w").write(res.stdout[-20000:])
            r["egitim_sn"] = round(time.time() - t); r["egitim_kod"] = res.returncode
            lora = out + "/pytorch_lora_weights.safetensors"
            if os.path.exists(lora):
                os.makedirs(OUT + "/modeller", exist_ok=True)
                subprocess.run("cp %s %s/modeller/tpn_%s.safetensors" % (lora, OUT, name), shell=True)
                t = time.time(); test_lora(name, lora, gpu); r["test_sn"] = round(time.time() - t)
        except Exception:
            r["hata"] = traceback.format_exc()[-2000:]
        json.dump(rep, open(OUT + "/rapor_gpu%d.json" % gpu, "w"), indent=1)

def to_hf_folder(d):
    """Görsel + açıklama çiftlerini 'imagefolder' biçimine çevirir (metadata.jsonl)."""
    import glob, shutil
    hf = d + "_hf"; os.makedirs(hf, exist_ok=True)
    with open(hf + "/metadata.jsonl", "w") as f:
        for p in sorted(glob.glob(d + "/*.png")):
            b = os.path.basename(p); shutil.copy(p, hf + "/" + b)
            f.write(json.dumps({"file_name": b, "text": open(p[:-4] + ".txt").read()}) + "\n")

def test_lora(name, lora, gpu):
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    sex, look = DESC[name]
    pipe = StableDiffusionXLPipeline.from_pretrained(BASE, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda:%d" % gpu)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.load_lora_weights(lora)
    os.makedirs(OUT + "/test", exist_ok=True)
    for i, scene in enumerate(["standing in a dark bathroom at night looking at a mirror, worried",
                               "sitting at a dinner table in a suburban house, talking, warm light",
                               "on a sunny suburban street in front of a white house, smiling"]):
        g = torch.Generator("cuda:%d" % gpu).manual_seed(7 + i)
        im = pipe(prompt="cinematic film still from a prestige thriller TV series, photo of tpn_%s %s, %s, %s, photorealistic, 35mm" % (name, sex, look, scene),
                  negative_prompt="cartoon, 3d, deformed, text, watermark, blurry", width=1216, height=704,
                  num_inference_steps=28, guidance_scale=5.0, generator=g).images[0]
        im.save(OUT + "/test/test_%s_%d.png" % (name, i + 1))
    del pipe; torch.cuda.empty_cache()

def main():
    os.makedirs(TMP, exist_ok=True)
    import diffusers
    ver = diffusers.__version__.split(".dev")[0]
    url = "https://raw.githubusercontent.com/huggingface/diffusers/v%s/examples/dreambooth/train_dreambooth_lora_sdxl.py" % ver
    try:
        urllib.request.urlretrieve(url, TMP + "/train_lora.py")
    except Exception:
        urllib.request.urlretrieve("https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py", TMP + "/train_lora.py")
    for name, u in JOB["cast"].items():
        urllib.request.urlretrieve(u, TMP + "/ref_%s.png" % name)
    names = list(JOB["cast"].keys())
    halves = [names[0::2], names[1::2]]
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    ps = [mp.Process(target=worker, args=(h, g)) for g, h in enumerate(halves) if h]
    for p in ps: p.start()
    for p in ps: p.join()

if __name__ == "__main__":
    t = time.time()
    try:
        main()
    except Exception:
        open(OUT + "/ana_hata.txt", "w").write(traceback.format_exc())
    open(OUT + "/toplam_sure.txt", "w").write("%.0f dakika" % ((time.time() - t) / 60))
