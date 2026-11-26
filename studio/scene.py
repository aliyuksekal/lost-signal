"""Lost Signal - Bölüm 1 teaser (30 sn). Kullanım:
python3 scene.py build            -> sahneyi kurar, ep1.blend kaydeder
python3 scene.py still <frame>    -> tek kare test
python3 scene.py render <a> <b>   -> a..b arası kareleri (ikişer adımla) render eder
"""
import bpy, math, random, sys, json, os
import numpy as np
from mathutils import Vector, Matrix, Euler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chars import clear, mat, rounded_box, cylinder, tube, ellipsoid, build_radio
import chars2
from chars2 import body, NORA, BENNY, KAI, fabric, NECK

FPS = 24
VERT = os.environ.get("VERT") == "1"
END = 660  # 27.5 sn, son 2.5 sn başlık kartı ffmpeg ile
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("LS_OUT", BASE)
HQ = os.environ.get("LS_QUALITY") == "hq"
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
LINES = json.load(open(BASE + "/audio/timeline.json"))  # [{"who":..,"file":..,"start":..}]

def f(t):  # saniye -> kare
    return int(round(t * FPS)) + 1

# ---------------------------------------------------------------- oda
def wood_mat(name, c1, c2, scale=12):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.45
    wv = nt.nodes.new("ShaderNodeTexWave"); wv.wave_type = "BANDS"; wv.bands_direction = "X"
    wv.inputs["Scale"].default_value = scale; wv.inputs["Distortion"].default_value = 8; wv.inputs["Detail"].default_value = 3
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c1, 1); ramp.color_ramp.elements[1].color = (*c2, 1)
    nt.links.new(wv.outputs["Fac"], ramp.inputs[0]); nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.1
    nt.links.new(wv.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m

def box(name, loc, size, m, bevel=0.01, rot=(0, 0, 0)):
    root = bpy.data.objects.get("SET")
    return rounded_box(name, loc, size, m, root, bevel=bevel, rot=rot)

def build_room():
    root = bpy.data.objects.new("SET", None); bpy.context.collection.objects.link(root)
    floor = wood_mat("floor", (0.22, 0.12, 0.06), (0.32, 0.18, 0.09), scale=3)
    wall = fabric("wall", (0.16, 0.24, 0.26), rough=0.9, bump_s=0.05, scale=40)
    box("floor", (0, 0.5, -0.02), (8, 6, 0.04), floor, 0.001)
    # arka duvar, pencere boşluklu
    WY = 1.9
    box("wall_l", (-2.35, WY, 1.5), (3.3, 0.1, 3.0), wall, 0.002)
    box("wall_r", (1.95, WY, 1.5), (2.9, 0.1, 3.0), wall, 0.002)
    box("wall_bot", (-0.1, WY, 0.45), (1.2, 0.1, 0.9), wall, 0.002)
    box("wall_top", (-0.1, WY, 2.55), (1.2, 0.1, 0.9), wall, 0.002)
    box("wall_side", (-2.4, 0.5, 1.5), (0.1, 6, 3.0), wall, 0.002)
    frame = mat("frame", (0.85, 0.83, 0.78), rough=0.4)
    for loc, size in (((-0.1, WY - 0.06, 0.92), (1.25, 0.08, 0.06)), ((-0.1, WY - 0.06, 2.08), (1.25, 0.08, 0.06)),
                      ((-0.69, WY - 0.06, 1.5), (0.06, 0.08, 1.2)), ((0.49, WY - 0.06, 1.5), (0.06, 0.08, 1.2)),
                      ((-0.1, WY - 0.06, 1.5), (0.03, 0.05, 1.15)), ((-0.1, WY - 0.06, 1.5), (1.15, 0.05, 0.03))):
        box("winframe", loc, size, frame, 0.004)
    box("ceiling", (0, 0.5, 2.75), (8, 6, 0.04), fabric("ceil", (0.2, 0.22, 0.24), rough=0.9), 0.001)
    box("sill", (-0.1, WY - 0.14, 0.88), (1.35, 0.22, 0.04), frame, 0.006)
    # gece gökyüzü
    sky = bpy.data.materials.new("sky"); sky.use_nodes = True
    nt = sky.node_tree; out = nt.nodes["Material Output"]; nt.nodes.remove(nt.nodes["Principled BSDF"])
    em = nt.nodes.new("ShaderNodeEmission"); grad = nt.nodes.new("ShaderNodeTexGradient")
    mapn = nt.nodes.new("ShaderNodeMapping"); tc = nt.nodes.new("ShaderNodeTexCoord")
    mapn.inputs["Rotation"].default_value = (0, math.radians(90), 0)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.004, 0.008, 0.03, 1); ramp.color_ramp.elements[1].color = (0.05, 0.08, 0.2, 1)
    nt.links.new(tc.outputs["Generated"], mapn.inputs[0]); nt.links.new(mapn.outputs[0], grad.inputs[0])
    nt.links.new(grad.outputs["Fac"], ramp.inputs[0]); nt.links.new(ramp.outputs[0], em.inputs["Color"])
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs[0], out.inputs[0])
    box("sky", (-0.1, WY + 1.5, 1.5), (6, 0.02, 4), sky, 0.001)
    ellipsoid("moon", (0.25, WY + 1.45, 1.85), (0.12, 0.01, 0.12), mat("moon", (0.9, 0.92, 1.0), emis=6), root)
    random.seed(11)
    star = mat("star", (1, 1, 1), emis=8)
    for i in range(40):
        ellipsoid("star", (random.uniform(-1.2, 1.0), WY + 1.44, random.uniform(1.0, 2.4)), (0.006,) * 3, star, root, seg=6, sub=0)
    # uzak kasaba ışıkları
    town = mat("town", (1.0, 0.7, 0.3), emis=5)
    for i in range(14):
        ellipsoid("townlight", (random.uniform(-1.0, 0.8), WY + 1.4, random.uniform(0.95, 1.05)), (0.008,) * 3, town, root, seg=6, sub=0)
    # posterler
    for loc, c in (((-1.5, WY - 0.06, 1.65), (0.6, 0.15, 0.12)), ((-1.0, WY - 0.06, 1.4), (0.85, 0.65, 0.2)), ((1.25, WY - 0.06, 1.6), (0.15, 0.35, 0.55))):
        box("poster", loc, (0.4, 0.01, 0.55), mat("poster%d" % int(loc[0] * 10), c, rough=0.7), 0.002)
    # raf + kitaplar
    shelf = wood_mat("shelfwood", (0.35, 0.2, 0.1), (0.45, 0.28, 0.14))
    box("shelf", (1.3, WY - 0.15, 1.15), (0.9, 0.22, 0.03), shelf, 0.005)
    random.seed(5)
    x = 0.92
    while x < 1.68:
        wdt = random.uniform(0.03, 0.06); h = random.uniform(0.18, 0.26)
        c = random.choice([(0.5, 0.1, 0.1), (0.1, 0.3, 0.5), (0.8, 0.6, 0.2), (0.2, 0.4, 0.2), (0.8, 0.8, 0.75)])
        box("book", (x + wdt / 2, WY - 0.15, 1.165 + h / 2), (wdt, 0.16, h), mat("book%d" % len(bpy.data.materials), c, rough=0.6), 0.004)
        x += wdt + 0.004
    # masa
    dw = wood_mat("deskwood", (0.3, 0.16, 0.07), (0.42, 0.25, 0.12), scale=6)
    box("desktop", (0, 0.42, 0.74), (1.6, 0.66, 0.04), dw, 0.01)
    for sx in (-0.75, 0.75):
        for sy in (0.14, 0.7):
            box("leg", (sx, sy, 0.36), (0.05, 0.05, 0.72), dw, 0.005)
    # masa lambası
    metal = mat("lampmetal", (0.12, 0.35, 0.3), rough=0.3, coat=0.5)
    cylinder("lampbase", (-0.68, 0.3, 0.775), 0.07, 0.025, metal, root)
    tube("lamparm", [(-0.68, 0.3, 0.78), (-0.7, 0.35, 1.0), (-0.58, 0.28, 1.08)], 0.012, metal, root)
    bm_cone = bpy.ops.mesh.primitive_cone_add(radius1=0.09, radius2=0.03, depth=0.12, location=(-0.55, 0.26, 1.03), rotation=(math.radians(20), math.radians(-30), 0))
    sh = bpy.context.active_object; sh.data.materials.append(metal); sh.parent = root
    sol = sh.modifiers.new("s", "SOLIDIFY"); sol.thickness = 0.005
    ellipsoid("bulb", (-0.54, 0.26, 1.0), (0.025,) * 3, mat("bulb", (1, 0.75, 0.45), emis=25), root)
    L = bpy.data.lights.new("lamp", "SPOT"); L.energy = 60; L.color = (1.0, 0.7, 0.4); L.spot_size = math.radians(110); L.shadow_soft_size = 0.05
    o = bpy.data.objects.new("lamp", L); o.location = (-0.54, 0.26, 0.99); bpy.context.collection.objects.link(o)
    o.rotation_euler = (Vector((0.1, 0.5, 0.7)) - o.location).to_track_quat("-Z", "Y").to_euler()
    # masa üstü eşyalar
    for i, c in enumerate(((0.6, 0.12, 0.1), (0.15, 0.3, 0.45), (0.85, 0.8, 0.7))):
        box("deskbook", (0.5, 0.5, 0.775 + i * 0.03), (0.28, 0.2, 0.028), mat("deskbook%d" % i, c, rough=0.6), 0.004, rot=(0, 0, 0.1 * i))
    cylinder("mug", (0.38, 0.28, 0.815), 0.04, 0.09, mat("mug", (0.9, 0.88, 0.82), rough=0.25, coat=0.6), root)
    # radyo
    build_radio(root, (0, 0.4, 0.84), 1.0)
    return root

def lights():
    def area(name, loc, target, energy, color, size):
        L = bpy.data.lights.new(name, "AREA"); L.energy = energy; L.color = color; L.size = size
        o = bpy.data.objects.new(name, L); o.location = loc; bpy.context.collection.objects.link(o)
        o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        return o
    area("moonlight", (0.3, 3.2, 2.6), (0, 0.8, 1.2), 140, (0.55, 0.68, 1.0), 1.2)
    area("fillcam", (0.5, -2.5, 1.8), (0, 0.8, 1.2), 40, (0.6, 0.7, 1.0), 3)
    area("bounce", (0, 0.4, 0.3), (0, 0.9, 1.4), 12, (1.0, 0.75, 0.5), 1.0)
    w = bpy.data.worlds.new("w"); bpy.context.scene.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.03, 0.06, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.5

# ---------------------------------------------------------------- ses zarfları
def envelopes():
    import wave
    env = {k: np.zeros(END + 30) for k in ("nora", "benny", "kai", "radio")}
    for ln in LINES:
        w = wave.open(BASE + "/audio/" + ln["file"] + ".wav")
        sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
        hop = sr // FPS
        for i in range(len(a) // hop):
            rms = np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2))
            fr = f(ln["start"]) + i
            if fr < len(env[ln["who"]]):
                env[ln["who"]][fr] = rms
    for k in env:
        e = env[k]; mx = e.max() or 1
        env[k] = np.clip(e / (mx * 0.6), 0, 1)
    return env

# ---------------------------------------------------------------- animasyon
def look_key(rig, target, frame, extra=(0, 0)):
    root = rig["root"]; bpy.context.view_layer.update()
    hc = root.matrix_world @ Vector((0, 0, NECK + 0.18))
    d = root.matrix_world.to_3x3().inverted() @ (Vector(target) - hc)
    yaw = math.atan2(d.x, -d.y); pitch = math.atan2(d.z, math.hypot(d.x, d.y))
    yaw = max(-0.9, min(0.9, yaw)); pitch = max(-0.5, min(0.4, pitch))
    p = rig["pivot"]
    p.rotation_euler = (-pitch * 0.8 + extra[0], extra[1], yaw * 0.85)
    p.keyframe_insert("rotation_euler", frame=frame)

def animate(rigs, env):
    sc = bpy.context.scene
    RADIO = (0, 0.4, 0.86)
    def head(n):
        r = rigs[n]["root"]; return r.matrix_world @ Vector((0, 0, NECK + 0.18))
    bpy.context.view_layer.update()
    plan = {
        "nora":  [(0, "R"), (4.0, "R"), (8.6, "benny"), (11.0, "benny"), (14.0, "kai"), (16.6, "kai"), (18.9, "R"), (24.5, "R")],
        "benny": [(0, "R"), (8.0, "kai"), (9.6, "R"), (11.8, "nora"), (16.8, "kai"), (18.9, "R"), (24.5, "R")],
        "kai":   [(0, "R"), (8.4, "benny"), (11.6, "nora"), (16.4, "nora"), (17.6, "R"), (18.9, "R"), (24.6, "CAM")],
    }
    for n, steps in plan.items():
        random.seed(hash(n) % 100)
        for i, (t, tgt) in enumerate(steps):
            if tgt == "R": p = RADIO
            elif tgt == "CAM": p = (0, -1.0, 1.55)
            else: p = head(tgt)
            look_key(rigs[n], p, f(t))
            # küçük yaşam hareketleri
            nt = steps[i + 1][0] if i + 1 < len(steps) else 27.5
            tt = t + 1.2
            while tt < nt - 0.6:
                look_key(rigs[n], p, f(tt), (random.uniform(-0.05, 0.05), random.uniform(-0.04, 0.04)))
                tt += random.uniform(1.0, 1.8)
            if i + 1 < len(steps):
                look_key(rigs[n], p, f(nt - 0.25))
    # göz kırpma
    for n, rig in rigs.items():
        random.seed(len(n) * 7)
        t = random.uniform(0.5, 2.0)
        while t < 27.5:
            for ob in rig["blink"]:
                ob.hide_render = True; ob.keyframe_insert("hide_render", frame=f(t) - 1)
                ob.hide_render = False; ob.keyframe_insert("hide_render", frame=f(t))
                ob.hide_render = True; ob.keyframe_insert("hide_render", frame=f(t) + 4)
            t += random.uniform(2.5, 4.5)
        # konuşma
        m = rig["mouth"]
        for fr in range(1, END + 2):
            v = env[n][fr] if fr < len(env[n]) else 0
            m.scale = (0.026 * (0.85 + 0.25 * v), 0.01, 0.001 + 0.016 * v)
            m.keyframe_insert("scale", frame=fr)
        # nefes / gövde salınımı
        r = rig["root"]; base = r.rotation_euler.z
        for i, t in enumerate(np.arange(0, 28, 2.0)):
            r.rotation_euler.z = base + math.radians(1.2) * math.sin(i * 1.7 + len(n))
            r.keyframe_insert("rotation_euler", frame=f(t))
    # radyo kadranı titremesi
    dial = bpy.data.materials["dial"].node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    glow = bpy.data.objects["dial_glow"].data
    random.seed(2)
    for fr in range(1, END + 2, 2):
        t = (fr - 1) / FPS
        e = env["radio"][fr] if fr < len(env["radio"]) else 0
        if t < 4: amp = 1.5 + 6 * random.random() * (t / 4)
        else: amp = 2 + 10 * e + 2 * random.random()
        dial.default_value = amp
        dial.keyframe_insert("default_value", frame=fr)
        glow.energy = 2 + amp * 1.5; glow.keyframe_insert("energy", frame=fr)

# ---------------------------------------------------------------- kameralar
def cameras(rigs):
    sc = bpy.context.scene
    def cam(name, keys, lens, fstop=2.8):
        c = bpy.data.cameras.new(name); c.lens = lens; c.dof.use_dof = True; c.dof.aperture_fstop = fstop
        o = bpy.data.objects.new(name, c); bpy.context.collection.objects.link(o)
        for fr, loc, tgt in keys:
            o.location = loc
            o.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
            c.dof.focus_distance = (Vector(tgt) - Vector(loc)).length
            o.keyframe_insert("location", frame=fr); o.keyframe_insert("rotation_euler", frame=fr)
            c.dof.keyframe_insert("focus_distance", frame=fr)
        return o
    def H(n, dz=0.18):
        r = rigs[n]["root"]; return tuple(r.matrix_world @ Vector((0, 0, NECK + dz)))
    bpy.context.view_layer.update()
    shots = [
        (0.0, cam("A_wide", [(f(0), (0.25, -2.1, 1.5), (0, 0.8, 1.2)), (f(3.5), (0.1, -1.7, 1.42), (0, 0.8, 1.15))], 30, 4)),
        (3.5, cam("B_radio", [(f(3.5), (0.22, -0.2, 0.95), (0, 0.4, 0.86)), (f(8.0), (0.12, -0.05, 0.92), (0, 0.4, 0.86))], 45, 2.0)),
        (8.0, cam("C_benny", [(f(8.0), (0.22, -0.4, 1.6), H("benny", 0.05)), (f(11.4), (0.26, -0.3, 1.6), H("benny", 0.05))], 45, 2.8)),
        (11.4, cam("D_nora", [(f(11.4), (-0.22, -0.4, 1.6), H("nora", 0.05)), (f(16.6), (-0.26, -0.3, 1.6), H("nora", 0.05))], 45, 2.8)),
        (16.6, cam("E_kai", [(f(16.6), (-0.15, -0.4, 1.62), H("kai", 0.08)), (f(18.8), (-0.12, -0.3, 1.62), H("kai", 0.08))], 45, 2.8)),
        (18.8, cam("F_push", [(f(18.8), (0.0, -1.9, 1.25), (0, 0.4, 0.95)), (f(24.4), (0.0, -0.6, 1.0), (0, 0.4, 0.88))], 32, 3.2)),
        (24.4, cam("G_kai", [(f(24.4), (0.0, -0.35, 1.6), H("kai", 0.12)), (f(27.5), (0.0, 0.1, 1.6), H("kai", 0.14))], 45, 2.4)),
    ]
    for t, o in shots:
        m = sc.timeline_markers.new(o.name, frame=f(t)); m.camera = o
    sc.camera = shots[0][1]
    if VERT:
        for t, o in shots:
            if o.name in ("A_wide", "F_push"):
                o.data.lens = 22

def enable_gpu():
    """Kaggle/GPU: OptiX, olmazsa CUDA. GPU yoksa CPU'da kalır."""
    sc = bpy.context.scene
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    sc.cycles.device = "GPU"
                    print("GPU:", kind, [d.name for d in gpus])
                    return True
            except Exception as e:
                print("GPU denemesi", kind, e)
    except Exception as e:
        print("GPU yok:", e)
    print("CPU ile render")
    return False

def setup_render(res=None, samples=None):
    if HQ:
        res = res or ((1080, 1920) if VERT else (1920, 1080))
        samples = samples or 128
    res = res or ((540, 960) if VERT else (854, 480))
    samples = samples or 12
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"
    sc.cycles.samples = samples; sc.cycles.use_denoising = True
    sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.02 if HQ else 0.05
    sc.cycles.max_bounces = 8 if HQ else 4
    sc.cycles.diffuse_bounces = 4 if HQ else 2; sc.cycles.glossy_bounces = 4 if HQ else 2
    sc.cycles.transmission_bounces = 2; sc.cycles.transparent_max_bounces = 4
    sc.render.use_persistent_data = True
    sc.render.resolution_x, sc.render.resolution_y = res; sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "AgX"
    try: sc.view_settings.look = "AgX - Medium High Contrast"
    except Exception: pass
    sc.render.fps = FPS; sc.frame_start = 1; sc.frame_end = END; sc.frame_step = 2
    sc.render.image_settings.file_format = "PNG"
    for ob in bpy.data.objects:
        for md in ob.modifiers:
            if md.type == "SUBSURF" and not HQ: md.render_levels = min(md.render_levels, 1)

def build():
    clear()
    build_room()
    rigs = {
        "nora": body(NORA, -0.62, 0.98, 22),
        "kai": body(KAI, 0.0, 0.92, 0, 1.04),
        "benny": body(BENNY, 0.64, 0.98, -22),
    }
    lights()
    animate(rigs, envelopes())
    cameras(rigs)
    setup_render()
    bpy.ops.wm.save_as_mainfile(filepath=OUT + ("/ep1_v.blend" if VERT else "/ep1.blend"))

if __name__ == "__main__":
    cmd = ARGS[0]
    if cmd == "build":
        build(); print("BUILT")
    else:
        bpy.ops.wm.open_mainfile(filepath=OUT + ("/ep1_v.blend" if VERT else "/ep1.blend"))
        sc = bpy.context.scene
        if os.environ.get("LS_GPU") == "1":
            enable_gpu()
        if cmd == "still":
            os.makedirs(OUT + "/renders", exist_ok=True)
            for fr in ARGS[1:]:
                sc.frame_set(int(fr)); sc.render.filepath = OUT + "/renders/%sstill_%s.png" % ("v" if VERT else "", fr)
                bpy.ops.render.render(write_still=True)
        elif cmd == "render":
            a, b = int(ARGS[1]), int(ARGS[2])
            os.makedirs(OUT + ("/frames_v" if VERT else "/frames"), exist_ok=True)
            for fr in range(a, b + 1, 2):
                out = OUT + ("/frames_v/%04d.png" if VERT else "/frames/%04d.png") % fr
                if os.path.exists(out): continue
                sc.frame_set(fr); sc.render.filepath = out
                bpy.ops.render.render(write_still=True)
        print("DONE")
