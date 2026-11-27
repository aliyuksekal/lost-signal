"""Scale Lab - "What If You Were the Last Person on Earth?" (full 3D look, Wan 2.2 TI2V-5B on Kaggle 2x T4).

Stages:
  1) encode : UMT5 prompt embeddings once (GPU0), saved to disk
  2) gen    : two workers (one per T4) generate 704x1280 clips with Wan 2.2 -> latents -> VAE decode -> mp4
  3) build  : compose 1080x1920 video (crop, retime to narration, captions, brand tag) + mux audio
Narration, music and word timings are prepared in ai/last_person/ (Kokoro TTS + synthesized score).
"""
import os, sys, json, time, gc, subprocess, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "last_person")
MID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
W, H = 704, 1280
FPS = 24
NEG = ("blurry, low quality, worst quality, jpeg artifacts, distorted face, deformed hands, extra fingers, extra limbs, "
       "morphing, melting, flicker, text, subtitles, watermark, logo, static frozen frame, oversaturated, ugly")


def shots():
    spec = json.load(open(A + "/spec.json"))
    out = []
    for si, s in enumerate(spec["segments"]):
        n = len(s["shots"])
        for k, p in enumerate(s["shots"]):
            prompt = p.replace("{hero}", spec["hero"]) + ". " + spec["style"]
            out.append({"id": "s%02d_%d" % (si, k), "seg": si, "prompt": prompt, "frames": 81 if n > 1 else 121})
    return out


def log(rep_path, key, val):
    try:
        r = json.load(open(rep_path))
    except Exception:
        r = {}
    r[key] = val
    json.dump(r, open(rep_path, "w"), indent=1)


# ---------------------------------------------------------------- 1) prompt embeddings
def encode(tmp):
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(MID, subfolder="tokenizer")
    for dtype in (torch.float16, torch.bfloat16):   # fp16 first (fast on T4); bf16 if fp16 overflows
        te = UMT5EncoderModel.from_pretrained(MID, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True).to("cuda:0")
        def emb(text):
            ids = tok([text], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
                      return_attention_mask=True, return_tensors="pt")
            mask = ids.attention_mask.to("cuda:0"); n = int(mask.sum())
            with torch.no_grad():
                e = te(ids.input_ids.to("cuda:0"), mask).last_hidden_state.float()
            if torch.isnan(e).any() or torch.isinf(e).any():
                raise RuntimeError("text encoder NaN")
            e = e[:, :n]
            e = torch.cat([e, e.new_zeros(1, 512 - n, e.shape[-1])], 1)  # same as diffusers: trim, zero-pad to 512
            return e.half().cpu()
        try:
            data = {"__neg__": emb(NEG)}
            for s in shots():
                data[s["id"]] = emb(s["prompt"])
            torch.save(data, tmp + "/emb.pt")
            return
        except RuntimeError as ex:
            if "NaN" not in str(ex):
                raise
        finally:
            del te; gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError("text encoder failed in fp16 and bf16")


# ---------------------------------------------------------------- 2) clip worker (one per GPU)
def worker(gpu, ids, tmp, rep):
    import torch, numpy as np
    from diffusers import WanPipeline, AutoencoderKLWan
    from diffusers.utils import export_to_video
    dev = "cuda:%d" % gpu
    rep = rep.replace(".json", "_gpu%d.json" % gpu)

    class Pipe(WanPipeline):  # VAE stays on CPU while the transformer runs on this GPU
        @property
        def _execution_device(self):
            return torch.device(dev)
    # stagger loading so both workers don't hold a full fp16 copy in CPU RAM at once
    if gpu == 1:
        t0 = time.time()
        while not os.path.exists(tmp + "/loaded_0") and time.time() - t0 < 1500:
            time.sleep(5)
    emb = torch.load(tmp + "/emb.pt")
    vae = AutoencoderKLWan.from_pretrained(MID, subfolder="vae", torch_dtype=torch.float32)
    pipe = Pipe.from_pretrained(MID, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=torch.float16)
    pipe.transformer.to(dev)
    open(tmp + "/loaded_%d" % gpu, "w").write("1")
    all_shots = {s["id"]: s for s in shots()}
    steps = 30
    lat = {}
    for sid in ids:
        s = all_shots[sid]
        t = time.time()
        try:
            g = torch.Generator(dev).manual_seed(1000 + sum(map(ord, sid)))
            out = pipe(prompt_embeds=emb[sid].to(dev), negative_prompt_embeds=emb["__neg__"].to(dev),
                       height=H, width=W, num_frames=s["frames"], num_inference_steps=steps, guidance_scale=5.0,
                       generator=g, output_type="latent").frames
            if torch.isnan(out).any():
                raise RuntimeError("NaN latents (fp16 overflow)")
            lat[sid] = out.cpu()
            dt = round(time.time() - t)
            log(rep, "gen_" + sid, dt)
            if dt > 2400 and steps > 20:   # too slow -> fewer steps for the rest
                steps = 20
        except Exception:
            log(rep, "gen_" + sid, "HATA " + traceback.format_exc()[-800:])
    # decode with the VAE only (free the transformer first)
    pipe.transformer = None; gc.collect(); torch.cuda.empty_cache()
    vae.to(dev)
    try:
        vae.enable_tiling()
    except Exception:
        pass
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(dev, torch.float32)
    std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(dev, torch.float32)
    for sid, l in lat.items():
        try:
            with torch.no_grad():
                v = vae.decode(l.to(dev, torch.float32) * std + mean, return_dict=False)[0]
            frames = pipe.video_processor.postprocess_video(v, output_type="np")[0]
            export_to_video(list(frames), tmp + "/clip_%s.mp4" % sid, fps=FPS)
            log(rep, "dec_" + sid, "ok")
        except Exception:
            log(rep, "dec_" + sid, "HATA " + traceback.format_exc()[-800:])
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- 3) compose
def load_frames(path):
    import imageio.v3 as iio
    return [f for f in iio.imiter(path)]


def build(tmp, out_dir, rep):
    import numpy as np
    import imageio_ffmpeg, shutil
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    ff = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    tm = json.load(open(A + "/timing.json")); words = json.load(open(A + "/words.json"))
    segs = tm["segments"]; total = tm["total"]
    sh = shots()
    # timeline: segment i covers [start_i, start_{i+1}) ; first starts at 0
    bounds = [0.0] + [s["start"] for s in segs[1:]] + [total]
    clips = {}
    for s in sh:
        p = tmp + "/clip_%s.mp4" % s["id"]
        if os.path.exists(p):
            clips[s["id"]] = p
    plan = []  # (t0, t1, clip path)
    last = None
    for si in range(len(segs)):
        ids = [s["id"] for s in sh if s["seg"] == si]
        have = [clips[i] for i in ids if i in clips] or ([last] if last else [])
        if not have:
            have = [None]
        t0, t1 = bounds[si], bounds[si + 1]
        step = (t1 - t0) / len(have)
        for k, p in enumerate(have):
            plan.append((t0 + k * step, t0 + (k + 1) * step, p))
        last = have[-1]
    log(rep, "plan", [(round(a, 2), round(b, 2), os.path.basename(p) if p else None) for a, b, p in plan])

    # caption chunks: up to 3 words, break after punctuation
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) == 3 or w["w"][-1:] in ".,?!":
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    for i, c in enumerate(chunks):
        c_end = chunks[i + 1][0]["s"] if i + 1 < len(chunks) and chunks[i + 1][0]["s"] - c[-1]["e"] < 0.6 else c[-1]["e"] + 0.25
        for w in c:
            w["chunk_end"] = c_end

    font = ImageFont.truetype(A + "/Anton-Regular.ttf", 92)
    tagf = ImageFont.truetype(A + "/Anton-Regular.ttf", 46)
    smallf = ImageFont.truetype(A + "/Anton-Regular.ttf", 34)
    OW, OH = 1080, 1920

    # static overlay: brand pill + vignette
    over = Image.new("RGBA", (OW, OH), (0, 0, 0, 0)); d = ImageDraw.Draw(over)
    tag = "WHAT IF…"
    tw = d.textlength(tag, font=tagf)
    d.rounded_rectangle(((OW - tw) / 2 - 34, 150, (OW + tw) / 2 + 34, 228), 39, fill=(255, 214, 10, 255))
    d.text(((OW - tw) / 2, 158), tag, font=tagf, fill=(15, 15, 15, 255))
    vig = Image.new("L", (OW, OH), 0); vd = ImageDraw.Draw(vig)
    vd.ellipse((-260, -200, OW + 260, OH + 200), fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(160))
    vig_arr = (np.asarray(vig, dtype=np.float32) / 255.0 * 0.38 + 0.62)[..., None]
    over_np = np.asarray(over, dtype=np.float32)
    o_rgb, o_a = over_np[..., :3], over_np[..., 3:] / 255.0

    def caption(img, t):
        act = [w for w in words if w["s"] <= t < w["chunk_end"]]
        if not act:
            return img
        w0 = act[-1]
        chunk = next(c for c in chunks if w0 in c)
        d = ImageDraw.Draw(img)
        txt = [w["w"].upper() for w in chunk]
        f = font
        for size in (92, 84, 76, 68, 60):
            f = ImageFont.truetype(A + "/Anton-Regular.ttf", size)
            widths = [d.textlength(x, font=f) for x in txt]
            sp = 22; total_w = sum(widths) + sp * (len(txt) - 1)
            if total_w <= 940:
                break
        x = (OW - total_w) / 2; y = 1180
        for w, x_txt, wd in zip(chunk, txt, widths):
            col = (255, 214, 10) if w is w0 else (255, 255, 255)
            d.text((x, y), x_txt, font=f, fill=col, stroke_width=9, stroke_fill=(0, 0, 0))
            x += wd + sp
        return img

    cache = {}
    def frames_of(p):
        if p not in cache:
            cache.clear(); cache[p] = load_frames(p)
        return cache[p]

    n_out = int(round(total * FPS))
    writer = subprocess.Popen([ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (OW, OH),
                               "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                               "-pix_fmt", "yuv420p", tmp + "/video.mp4"], stdin=subprocess.PIPE)
    for n in range(n_out):
        t = n / FPS
        seg_i = next(i for i, (a, b, p) in enumerate(plan) if a <= t < b or i == len(plan) - 1)
        a, b, p = plan[seg_i]
        if p is None:
            fr = np.zeros((H, W, 3), np.uint8)
        else:
            fs = frames_of(p)
            # map shot time to clip time; slow down if the slot is longer than the clip (blend neighbours)
            u = (t - a) / max(1e-6, b - a)
            span = min(len(fs) - 1, (b - a) * FPS)  # play at normal speed if clip is long enough
            pos = u * span
            i0 = int(pos); fr_ = pos - i0; i1 = min(i0 + 1, len(fs) - 1)
            fr = fs[i0] if fr_ < 0.05 else (fs[i0].astype(np.float32) * (1 - fr_) + fs[i1].astype(np.float32) * fr_).astype(np.uint8)
        im = Image.fromarray(fr).resize((OW, int(round(OW * fr.shape[0] / fr.shape[1]))), Image.LANCZOS)
        top = (im.height - OH) // 2
        im = im.crop((0, top, OW, top + OH)) if im.height >= OH else im.resize((OW, OH), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32)
        arr = np.clip((arr - 128) * 1.05 + 128, 0, 255) * vig_arr          # light contrast + vignette
        arr = arr * (1 - o_a) + o_rgb * o_a
        if t < 0.5:
            arr *= t / 0.5
        if t > total - 0.6:
            arr *= max(0.0, (total - t) / 0.6)
        im = Image.fromarray(arr.astype(np.uint8))
        if t < 3.6:  # thought-experiment label at the start
            dd = ImageDraw.Draw(im); lab = "A THOUGHT EXPERIMENT"
            dd.text(((OW - dd.textlength(lab, font=smallf)) / 2, 250), lab, font=smallf, fill=(235, 235, 235), stroke_width=4, stroke_fill=(0, 0, 0))
        im = caption(im, t)
        writer.stdin.write(im.tobytes())
        if n == int(1.5 * FPS):
            im.save(out_dir + "/kapak_1.5sn.jpg", quality=92)
    writer.stdin.close(); writer.wait()
    final = out_dir + "/LastPersonOnEarth_ScaleLab.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-i", tmp + "/video.mp4", "-i", A + "/audio.flac", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", final], check=True)
    log(rep, "final", os.path.basename(final))


def main(out_dir):
    tmp = "/tmp/lp"; os.makedirs(tmp, exist_ok=True)
    rep = out_dir + "/report.json"
    t = time.time()
    try:
        encode(tmp); log(rep, "encode_sn", round(time.time() - t))
    except Exception:
        log(rep, "encode_hata", traceback.format_exc()[-2500:])
    import torch
    ng = max(1, torch.cuda.device_count())
    sh = sorted(shots(), key=lambda s: -s["frames"])
    groups = [[] for _ in range(ng)]; cost = [0] * ng
    for s in sh:  # balance work across GPUs
        g = cost.index(min(cost)); groups[g].append(s["id"]); cost[g] += s["frames"]
    if ng == 1:
        open(tmp + "/loaded_0", "w").write("1")
    procs = []
    for g in range(ng):
        procs.append(subprocess.Popen([sys.executable, __file__, "--gen", str(g), ",".join(groups[g]), tmp, rep],
                                      stdout=open(out_dir + "/gpu%d_log.txt" % g, "w"), stderr=subprocess.STDOUT))
        time.sleep(2)
    for p in procs:
        p.wait()
    log(rep, "gen_toplam_sn", round(time.time() - t))
    try:
        build(tmp, out_dir, rep)
    except Exception:
        log(rep, "build_hata", traceback.format_exc()[-2500:])
    log(rep, "toplam_sn", round(time.time() - t))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--gen":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        worker(int(sys.argv[2]), [x for x in sys.argv[3].split(",") if x], sys.argv[4], sys.argv[5])
    elif len(sys.argv) > 1 and sys.argv[1] == "--build-only":
        build(sys.argv[2], sys.argv[3], sys.argv[3] + "/report.json")
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working")
