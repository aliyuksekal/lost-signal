"""The Perfect Neighbors - oyuncu seçmesi: her karakter için 4 yüz adayı + numaralı karşılaştırma sayfası."""
import os, sys, json, time, traceback
OUT = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working"
REPORT = {}

STYLE = ("cinematic head and shoulders portrait, prestige TV drama casting photo, photorealistic, 85mm lens, soft natural window light, "
         "neutral muted background, highly detailed natural skin texture, looking at camera, neutral expression")
NEG = "cartoon, anime, 3d render, cgi, painting, illustration, deformed face, asymmetric eyes, text, watermark, logo, blurry, oversaturated, heavy makeup"
CAST = {
    "emma":    "32-year-old woman, shoulder-length dark brown wavy hair, hazel eyes, light olive skin, minimal makeup, cream knit sweater, thoughtful intense gaze",
    "daniel":  "35-year-old man, short neat light brown hair, light stubble, blue eyes, fair skin, navy button-up shirt, warm but tired face",
    "maggie":  "64-year-old woman, silver hair in a neat bob, pearl earrings, pastel cardigan, polite smile with cold calculating eyes",
    "grant":   "46-year-old man, tall, salt-and-pepper hair swept back, tanned skin, grey polo shirt, confident charming smile",
    "vanessa": "41-year-old woman, long sleek platinum blonde hair, high cheekbones, white silk blouse, sad eyes behind a perfect smile",
    "owen":    "38-year-old man, thin face, messy dark hair, thick black-rimmed glasses, pale skin, grey hoodie, nervous polite smile",
    "rosa":    "58-year-old Latina woman, grey-streaked black hair in a low bun, reading glasses on a chain, denim shirt, sharp knowing eyes",
    "claire":  "34-year-old woman, short platinum pixie cut, pale blue eyes, small scar through her left eyebrow, black turtleneck, haunted expression",
}
SEEDS = [101, 202, 303, 404]

def main():
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    from PIL import Image, ImageDraw, ImageFont
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V5.0", torch_dtype=torch.float16, variant="fp16",
                                                     use_safetensors=True).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    REPORT["yukleme_sn"] = round(time.time() - t0, 1)
    try:
        import matplotlib
        fnt = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans-Bold.ttf"), 44)
    except Exception:
        fnt = ImageFont.load_default()
    for name, desc in CAST.items():
        t = time.time(); tiles = []
        os.makedirs(OUT + "/adaylar", exist_ok=True)
        for i, sd in enumerate(SEEDS, 1):
            g = torch.Generator("cuda").manual_seed(sd)
            im = pipe(prompt=desc + ", " + STYLE, negative_prompt=NEG, width=832, height=1216,
                      num_inference_steps=28, guidance_scale=5.0, generator=g).images[0]
            im.save(OUT + "/adaylar/%s_%d.png" % (name, i))
            tiles.append(im.resize((416, 608)))
        sheet = Image.new("RGB", (416 * 4, 608 + 70), (20, 20, 24)); d = ImageDraw.Draw(sheet)
        for i, tl in enumerate(tiles):
            sheet.paste(tl, (i * 416, 70))
            d.text((i * 416 + 14, 10), "%s %d" % (name.upper(), i + 1), font=fnt, fill=(255, 220, 120))
        sheet.save(OUT + "/secim_%s.jpg" % name, quality=88)
        REPORT[name] = round(time.time() - t, 1)
        json.dump(REPORT, open(OUT + "/report.json", "w"), indent=1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        REPORT["hata"] = traceback.format_exc()[-2500:]
    json.dump(REPORT, open(OUT + "/report.json", "w"), indent=1)
    print(REPORT)
