"""Kareleri videoya çevirir: altyazı (PIL ile kareye basılır), başlık kartı, ses.
Kullanım: python3 finish.py <kare_klasoru> <ses.wav> <altyazi.srt> <cikti.mp4> [alt_baslik]
"""
import os, sys, re, glob, shutil, subprocess, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FPS_IN = 12          # ikişer adımla render (24 fps'nin yarısı)
TITLE_SEC = 2.5

def font(bold, size):
    cands = []
    try:
        import matplotlib
        d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        cands.append(os.path.join(d, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
    except Exception:
        pass
    cands += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

def ffmpeg_bin():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def parse_srt(path):
    out = []
    txt = open(path, encoding="utf-8").read().strip()
    for block in re.split(r"\n\s*\n", txt):
        lines = block.strip().splitlines()
        if len(lines) < 3: continue
        m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1])
        if not m: continue
        g = [int(x) for x in m.groups()]
        a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        out.append((a, b, " ".join(lines[2:])))
    return out

def wrap(draw, text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= maxw: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def draw_sub(im, text):
    W, H = im.size
    d = ImageDraw.Draw(im)
    vertical = H > W
    size = int(W * (0.055 if vertical else 0.034))
    fnt = font(True, size)
    lines = wrap(d, text, fnt, W * 0.86)
    lh = int(size * 1.25)
    y0 = int(H * (0.62 if vertical else 0.86)) - lh * (len(lines) - 1)
    for i, ln in enumerate(lines):
        w = d.textlength(ln, font=fnt); x = (W - w) / 2; y = y0 + i * lh
        d.text((x, y), ln, font=fnt, fill=(255, 255, 255), stroke_width=max(2, size // 12), stroke_fill=(0, 0, 0))

def title_card(W, H, sub):
    random.seed(3)
    im = Image.new("RGB", (W, H), (4, 6, 12)); d = ImageDraw.Draw(im)
    for _ in range(int(W * H / 350)):
        x, y = random.randint(0, W - 1), random.randint(0, H - 1); g = random.randint(10, 40)
        d.point((x, y), (g, g, g + 10))
    vertical = H > W
    f = font(True, int(W * (0.14 if vertical else 0.1)))
    lines = ["LOST", "SIGNAL"] if vertical else ["LOST SIGNAL"]
    lh = int(f.size * 1.15); y0 = int(H * 0.42) - lh * len(lines) // 2
    glow = Image.new("RGB", (W, H)); gd = ImageDraw.Draw(glow)
    for i, t in enumerate(lines):
        gd.text(((W - d.textlength(t, font=f)) / 2, y0 + i * lh), t, font=f, fill=(255, 150, 40))
    glow = glow.filter(ImageFilter.GaussianBlur(W // 50))
    im = Image.composite(glow, im, glow.convert("L")); d = ImageDraw.Draw(im)
    o = max(3, W // 200)
    for i, t in enumerate(lines):
        x = (W - d.textlength(t, font=f)) / 2; y = y0 + i * lh
        d.text((x - o, y), t, font=f, fill=(40, 180, 255)); d.text((x + o, y), t, font=f, fill=(255, 60, 60))
        d.text((x, y), t, font=f, fill=(255, 236, 205))
    f2 = font(False, int(W * (0.043 if vertical else 0.024)))
    for j, s in enumerate([sub, "Subscribe!"]):
        d.text(((W - d.textlength(s, font=f2)) / 2, y0 + len(lines) * lh + int(f2.size * 1.6 * (j + 0.6))), s, font=f2, fill=(210, 210, 220))
    for yy in range(0, H, 4): d.line((0, yy, W, yy), fill=(0, 0, 0))
    return im

def main():
    frames_dir, wav, srt, out = sys.argv[1:5]
    sub = sys.argv[5] if len(sys.argv) > 5 else "New episodes Mon · Wed · Fri"
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    tmp = out + "_seq"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    subs = parse_srt(srt)
    W, H = Image.open(frames[0]).size
    n = 0
    for i, fp in enumerate(frames):
        t = i / FPS_IN
        im = Image.open(fp).convert("RGB")
        for a, b, text in subs:
            if a <= t < b: draw_sub(im, text)
        im.save(os.path.join(tmp, "%05d.png" % n)); n += 1
    card = title_card(W, H, sub)
    black = Image.new("RGB", (W, H))
    k = int(TITLE_SEC * FPS_IN)
    for j in range(k):
        Image.blend(black, card, min(1.0, j / 5)).save(os.path.join(tmp, "%05d.png" % n)); n += 1
    ff = ffmpeg_bin()
    subprocess.run([ff, "-y", "-v", "error", "-framerate", str(FPS_IN), "-i", os.path.join(tmp, "%05d.png"), "-i", wav,
                    "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("VIDEO", out)

if __name__ == "__main__":
    main()
