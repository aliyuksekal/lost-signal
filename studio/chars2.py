"""Lost Signal - karakterler v2 (daha gerçekçi oranlar, parmaklı eller,
cilt/kumaş dokusu, hareket ettirilebilir kafa, göz kırpma ve konuşan ağız)."""
import bpy, bmesh, math, random
from mathutils import Vector, Quaternion, Matrix
from chars import (mat, mat_stripes, ellipsoid, shell, tube, rounded_box, cylinder,
                   Head, new_obj, smooth, link, face_curve, build_radio, _mats)

HZ = 1.45
HR = (0.15, 0.145, 0.168)
NECK = 1.27
K = 0.15 / 0.215  # v1'e göre kafa ölçeği

# ---------------------------------------------------------------- malzemeler
def skin_mat(name, color):
    if name in _mats: return _mats[name]
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.48
    b.inputs["Subsurface Weight"].default_value = 0.35
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.3, 0.15)
    b.inputs["Subsurface Scale"].default_value = 0.025
    b.inputs["Specular IOR Level"].default_value = 0.45
    noise = nt.nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 60
    noise.inputs["Detail"].default_value = 6
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = tuple(c * 0.9 for c in color) + (1,)
    ramp.color_ramp.elements[1].color = (min(1, color[0] * 1.05), color[1] * 1.0, color[2] * 0.97, 1)
    ramp.color_ramp.elements[0].position = 0.35; ramp.color_ramp.elements[1].position = 0.65
    nt.links.new(noise.outputs["Fac"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
    pores = nt.nodes.new("ShaderNodeTexNoise"); pores.inputs["Scale"].default_value = 900
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.06
    nt.links.new(pores.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    _mats[name] = m
    return m

def fabric(name, color, rough=0.85, bump_s=0.25, scale=220, base=None):
    if name in _mats: return _mats[name]
    m = base or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    if color is not None:
        b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Sheen Weight"].default_value = 0.4
    tex = nt.nodes.new("ShaderNodeTexWave"); tex.wave_type = "BANDS"
    tex.inputs["Scale"].default_value = scale; tex.inputs["Distortion"].default_value = 0.5
    folds = nt.nodes.new("ShaderNodeTexNoise"); folds.inputs["Scale"].default_value = 6
    mix = nt.nodes.new("ShaderNodeMath"); mix.operation = "ADD"
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 4.0
    nt.links.new(folds.outputs["Fac"], mul.inputs[0])
    nt.links.new(tex.outputs["Fac"], mix.inputs[0]); nt.links.new(mul.outputs[0], mix.inputs[1])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = bump_s
    bump.inputs["Distance"].default_value = 0.02
    nt.links.new(mix.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    _mats[name] = m
    return m

def hair_mat2(name, color):
    if name in _mats: return _mats[name]
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = 0.5
    b.inputs["Coat Weight"].default_value = 0.05
    b.inputs["Anisotropic"].default_value = 0.6
    w = nt.nodes.new("ShaderNodeTexWave"); w.wave_type = "BANDS"; w.bands_direction = "X"
    w.inputs["Scale"].default_value = 60; w.inputs["Distortion"].default_value = 6
    w.inputs["Detail"].default_value = 4
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Object"], w.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.5
    nt.links.new(w.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = tuple(c * 0.6 for c in color) + (1,)
    ramp.color_ramp.elements[1].color = tuple(min(1, c * 1.3) for c in color) + (1,)
    nt.links.new(w.outputs["Fac"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
    _mats[name] = m
    return m


# ---------------------------------------------------------------- tek parça gövde ve kafa
def _interp(pts, t):
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            k = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            k = k * k * (3 - 2 * k)
            return v0 + (v1 - v0) * k
    return pts[-1][1]

def torso_mesh(name, z0, z1, prof_x, prof_y, m, root, belly=0.0, flare=0.0):
    """Tek parça, dikişsiz gövde: süper-elips dikey profil, yükseklikte genişlik eğrileri."""
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=56, v_segments=36, radius=1.0)
    for v in bm.verts:
        z = max(-1.0, min(1.0, v.co.z)); t = (z + 1) / 2
        r0 = math.sqrt(max(1e-9, v.co.x ** 2 + v.co.y ** 2))
        dx, dy = (v.co.x / r0, v.co.y / r0) if r0 > 1e-6 else (0.0, 0.0)
        s = (1 - abs(z) ** 4) ** 0.25 if abs(z) < 1 else 0.0
        rx = _interp(prof_x, t) * (1 + flare * max(0, 0.35 - t))
        ry = _interp(prof_y, t) * (1 + flare * max(0, 0.35 - t))
        if dy < 0 and belly:
            ry *= 1 + belly * math.exp(-((t - 0.32) / 0.2) ** 2) * (-dy)
        v.co = Vector((dx * rx * s, dy * ry * s, z0 + t * (z1 - z0)))
    ob = new_obj(name, bm, m, root)
    return smooth(ob, 2)

def head_mesh(head, skin, H, jaw=1.0):
    """Tek parça kafa: çene incelir, yanaklar dolgun, çene ucu öne çıkar. Dikiş yok."""
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=40, radius=1.0)
    for v in bm.verts:
        x, y, z = v.co
        low = max(0.0, -z)
        x *= 1 - 0.2 * jaw * low ** 1.7                     # çene hattı
        if y < 0:
            y *= 1 + 0.06 * math.exp(-((z + 0.3) / 0.25) ** 2) * (-y)   # yanak dolgunluğu
            y *= 1 + 0.08 * math.exp(-((z + 0.85) / 0.15) ** 2) * max(0, -y - 0.3)  # çene ucu
        if z > 0.3:
            x *= 1 + 0.03 * (z - 0.3)                        # alın/kafatası
        v.co = Vector((x * head.r.x, y * head.r.y, z * head.r.z * (1 + 0.06 * low))) + head.c
    ob = new_obj("head", bm, skin, H)
    return smooth(ob, 1)

# ---------------------------------------------------------------- yüz
def build_eye(tag, head, yaw, pitch, iris_col, lid, lid_tilt, skin, parent, rig, look=Vector((0, -6, 1.45)), size=0.034):
    n = head.normal(yaw, pitch)
    ec = head.point(yaw, pitch, -size * 0.42)
    white = mat("sclera2", (0.92, 0.9, 0.88), rough=0.12, coat=1.0)
    ellipsoid(tag + "_eye", ec, (size,) * 3, white, parent, sub=1)
    d = (look - ec).normalized()
    q = d.to_track_quat("-Y", "Z")
    irm = mat("iris2_" + tag, iris_col, rough=0.2, coat=1.0)
    ellipsoid(tag + "_iris", ec + d * (size - 0.004), (size * 0.6, 0.007, size * 0.6), irm, parent, rot=q)
    ellipsoid(tag + "_pupil", ec + d * (size + 0.0005), (size * 0.27, 0.005, size * 0.27), mat("pupil", (0.01, 0.01, 0.012), rough=0.2, coat=1), parent, rot=q)
    right = d.cross(Vector((0, 0, 1))).normalized(); up = right.cross(d).normalized()
    hl = ec + d * (size + 0.003) + up * size * 0.3 - right * size * 0.25
    ellipsoid(tag + "_hl", hl, (size * 0.1, 0.002, size * 0.1), mat("catch", (1, 1, 1), emis=5.0), parent, rot=q)
    # üst göz kapağı
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < lid], context="VERTS")
    ob = new_obj(tag + "_lid", bm, skin, parent)
    ob.location = ec; ob.scale = (size * 1.08,) * 3
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = n.to_track_quat("-Y", "Z") @ Quaternion((0, 1, 0), math.radians(lid_tilt))
    s = ob.modifiers.new("solid", "SOLIDIFY"); s.thickness = 0.1
    smooth(ob, 2)
    # kirpik çizgisi (kapak kenarı)
    lash = mat("lash", (0.03, 0.02, 0.02), rough=0.5)
    qq = ob.rotation_quaternion
    pts = []
    zl = lid
    rr = math.sqrt(max(0.0, 1 - zl * zl))
    for a in (-80, -40, 0, 40, 80):
        ar = math.radians(a)
        local = Vector((math.sin(ar) * rr, -math.cos(ar) * rr, zl)) * size * 1.1
        pts.append(ec + qq @ local)
    tube(tag + "_lash", pts, 0.0028 if "nora" not in tag else 0.0038, lash, parent)
    # kapalı göz (kırpma için)
    cl = ellipsoid(tag + "_closed", ec, (size * 1.1,) * 3, skin, parent, sub=1)
    cl.hide_render = True
    rig["blink"].append(cl)
    rig["lids"].append(ob)

def build_hand(tag, pts, skin, parent, w=1.0, grip=0.3):
    wr = Vector(pts[-1]); d = (wr - Vector(pts[-2])).normalized()
    side = d.cross(Vector((0, -1, 0)))
    if side.length < 0.1: side = Vector((1, 0, 0))
    side.normalize()
    front = side.cross(d).normalized()
    palm_c = wr + d * 0.045
    q = Matrix((side, front, d)).transposed().to_quaternion()
    ellipsoid(tag + "_palm", palm_c, (0.036 * w, 0.017 * w, 0.042), skin, parent, rot=q)
    for i, o in enumerate((-1.5, -0.5, 0.5, 1.5)):
        base = palm_c + d * 0.035 + side * o * 0.017 * w
        ln = (0.045, 0.052, 0.05, 0.04)[i]
        mid = base + d * ln * 0.55 + front * ln * grip * 0.4
        tip = base + d * ln * (1 - grip * 0.3) + front * ln * grip
        tube(tag + "_f", [base, mid, tip], 0.0085 * w, skin, parent, taper=[1.0, 0.9, 0.8])
    tb = palm_c + front * 0.015 - side * 0.03 * w
    tube(tag + "_thumb", [tb, tb + front * 0.02 - side * 0.012 + d * 0.015, tb + front * 0.032 + d * 0.035], 0.01 * w, skin, parent, taper=[1.0, 0.9, 0.8])

def body(P, x=0.0, y=0.0, yaw=0.0, scale=1.0):
    name = P["name"]; w = P.get("w", 1.0)
    root = bpy.data.objects.new(name, None); bpy.context.collection.objects.link(root)
    pivot = bpy.data.objects.new(name + "_headpivot", None); bpy.context.collection.objects.link(pivot)
    pivot.parent = root; pivot.location = (0, 0, NECK)
    rig = dict(root=root, pivot=pivot, blink=[], lids=[], mouth=None)
    skin = skin_mat("skin2_" + name, P["skin"])
    hairm = hair_mat2("hair2_" + name, P["hair"])
    pants = fabric("pants2_" + name, P["pants"], scale=300)
    shoe = mat("shoe2_" + name, P["shoe"], rough=0.45, coat=0.3)
    sole = mat("sole", (0.92, 0.9, 0.86), rough=0.6)
    top = P["top"]()
    head = Head((0, 0, HZ), HR)

    # ---- kafa (pivot'a bağlı)
    H = pivot
    head_mesh(head, skin, H, P.get("jaw", 1.0))
    ellipsoid("nose", head.point(0, -6, 0.006), (0.019, 0.03, 0.03), skin, H, rot=head.facing(0, -6))
    ellipsoid("nosetip", head.point(0, -12, 0.018), (0.02, 0.02, 0.018), skin, H, rot=head.facing(0, -12))
    for s in (-1, 1):
        ellipsoid("nostril", head.point(7 * s, -13, 0.008), (0.011, 0.011, 0.009), skin, H)
    if P.get("ears", True):
        for s in (-1, 1):
            ellipsoid("ear", head.point(88 * s, -3, -0.006), (0.018, 0.03, 0.04), skin, H, rot=head.facing(88 * s, -3))
    for s in (-1, 1):
        build_eye(name + "_e%d" % s, head, 22 * s, 3, P["eyes"], P["lid"][0 if s < 0 else 1], P["lid_tilt"][0 if s < 0 else 1] * s, skin, H, rig)
    browm = mat("brow2_" + name, tuple(c * 0.8 for c in P["hair"]), rough=0.7)
    for s in (-1, 1):
        r_ = P["brows"][0 if s < 0 else 1]
        face_curve("brow", head, [(10 * s, 17 + r_[0]), (21 * s, 20 + r_[1]), (33 * s, 17 + r_[2])], 0.005, 0.0075, browm, H, taper=[1.0, 1.1, 0.6])
    # ağız: dudak çizgisi + konuşurken açılan iç ağız
    lip = mat("lip2_" + name, P["lip"], rough=0.4, coat=0.3)
    face_curve("mouthline", head, P["mouth"], 0.003, 0.0055, lip, H)
    ellipsoid("lowerlip", head.point(1, -31, 0.0), (0.024, 0.008, 0.009), lip, H, rot=head.facing(1, -31))
    inner = ellipsoid("mouthin", head.point(0, -27, -0.002), (0.03, 0.01, 0.001), mat("mouthin", (0.22, 0.03, 0.05), rough=0.5), H, rot=head.facing(0, -27))
    rig["mouth"] = inner
    blush = mat("blush2", (0.9, 0.42, 0.4), rough=0.6, alpha=0.22)
    for s in (-1, 1):
        ellipsoid("blush", head.point(34 * s, -12, 0.001), (0.028, 0.003, 0.018), blush, H, rot=head.facing(34 * s, -12))
    P["hair_fn"](head, hairm, H)
    if "face_extra" in P: P["face_extra"](head, H)
    ellipsoid("neck", (0, 0.008, 1.25), (0.048, 0.045, 0.08), skin, root)
    # pivot uzayına çevir
    inv = Matrix.Translation((0, 0, -NECK))
    for ob in pivot.children:
        ob.matrix_parent_inverse = inv

    # ---- gövde
    P["torso_fn"](top, root, w)
    ellipsoid("hips", (0, 0, 0.8), (0.145 * w, 0.1 * w, 0.1), pants, root)
    for s in (-1, 1):
        tube("leg", [(0.075 * s * w, 0, 0.82), (0.082 * s * w, -0.012, 0.46), (0.088 * s * w, 0, 0.1)], 0.058 * P.get("legw", 1.0), pants, root, taper=[1.1, 0.85, 0.7])
        ellipsoid("shoe", (0.088 * s * w, -0.045, 0.05), (0.055, 0.115, 0.048), shoe, root, sub=2)
        ellipsoid("sole", (0.088 * s * w, -0.045, 0.018), (0.062, 0.122, 0.02), sole, root, sub=2)
    sleeve = P.get("sleeve", top)
    for side, pts in (("L", P["armL"]), ("R", P["armR"])):
        pts = [Vector(p) for p in pts]
        # omuz: kol gövdenin içinden başlasın (dikişsiz omuz)
        pts[0] = Vector((math.copysign(0.135 * w, pts[0].x), pts[0].y, 1.13))
        sh = pts[0] - Vector((math.copysign(0.025, pts[0].x), 0, 0.008))
        tube("arm" + side, pts, 0.044 * P.get("armw", 1.0), sleeve, root, taper=[1.0, 0.85, 0.72])
        ellipsoid("cuff" + side, pts[-1], (0.034, 0.034, 0.012), sleeve, root, rot=(pts[-1] - pts[-2]).to_track_quat("Z", "Y"))
        build_hand(name + side, pts, skin, root, P.get("handw", 1.0), P.get("grip", 0.3))
    if "body_extra" in P: P["body_extra"](root, w)
    root.location = (x, y, 0); root.rotation_euler = (0, 0, math.radians(yaw)); root.scale = (scale,) * 3
    rig["head"] = head
    return rig

# ---------------------------------------------------------------- NORA
def nora_hair(head, m, H):
    def keep(d):
        if d.y < -0.22 and d.z < 0.2: return False
        return d.z > -0.62
    shell("hair_bob", head.c + Vector((0, 0.008, 0.0)), head.r * 1.09, keep, m, H, thick=0.022)
    for s in (-1, 1):
        ellipsoid("bob_tip", head.c + Vector((0.14 * s, 0.015, -0.12)), (0.042, 0.09, 0.035), m, H, rot=(0, math.radians(25 * s), 0))
    for yaw in range(-42, 43, 7):
        top = head.point(yaw * 0.6, 55, 0.024); mid = head.point(yaw * 0.9, 34, 0.021)
        end = head.point(yaw + 3, 16 + (yaw % 3), 0.016)
        tube("fringe", [top, mid, end], 0.017, m, H, taper=[1.0, 0.9, 0.4])
    # yanlarda tutamlar
    for s in (-1, 1):
        for i in range(5):
            yaw = (60 + i * 9) * s
            tube("side", [head.point(yaw, 40, 0.025), head.point(yaw + 5 * s, 0, 0.03), head.point(yaw + 2 * s, -32, 0.022)], 0.02, m, H, taper=[1, 1, 0.5])
    ellipsoid("bob_back", head.c + Vector((0, 0.07, -0.1)), (0.14, 0.07, 0.05), m, H)

def nora_face(head, H):
    fr = mat("freckle", (0.5, 0.22, 0.1), rough=0.6)
    random.seed(4)
    for s in (-1, 1):
        for i in range(10):
            y = 28 + random.uniform(-9, 9); p = -8 + random.uniform(-5, 5)
            ellipsoid("freckle", head.point(y * s, p, 0.0008), (0.0035, 0.001, 0.0035), fr, H, rot=head.facing(y * s, p), seg=8, sub=0)
    tube("pencil", [head.point(80, 14, 0.022), head.point(72, 26, 0.05)], 0.005, mat("pencil", (0.95, 0.7, 0.1)), H)

def nora_torso(m, root, w):
    torso_mesh("raincoat", 0.66, 1.25,
               [(0, 0.215), (0.3, 0.185), (0.6, 0.16), (0.82, 0.18), (0.92, 0.17), (1, 0.06)],
               [(0, 0.15), (0.3, 0.125), (0.6, 0.115), (0.85, 0.115), (1, 0.05)], m, root)
    collar = mat("coat_dark2", (0.6, 0.34, 0.03), rough=0.35, coat=0.6)
    for s in (-1, 1):
        ellipsoid("collar", (0.055 * s, -0.07, 1.215), (0.055, 0.022, 0.035), collar, root, rot=(0.3, 0, -0.5 * s))
    btn = mat("button", (0.25, 0.14, 0.06), rough=0.3, coat=1)
    for z in (1.12, 1.0, 0.88, 0.76):
        tt = (z - 0.66) / 0.59; zz = 2 * tt - 1
        yy = -_interp([(0, 0.15), (0.3, 0.125), (0.6, 0.115), (0.85, 0.115), (1, 0.05)], tt) * (1 - abs(zz) ** 4) ** 0.25
        ellipsoid("btn", (0.0, yy - 0.003, z), (0.012, 0.006, 0.012), btn, root)
    book = mat("notebook", (0.65, 0.07, 0.07), rough=0.5)
    rounded_box("notebook", (0.13, -0.19, 1.0), (0.09, 0.02, 0.12), book, root, bevel=0.005, rot=(0.25, 0, -0.25))

def nora_top():
    m = bpy.data.materials.new("raincoat2")
    fabric("raincoat2", (0.93, 0.58, 0.03), rough=0.28, bump_s=0.08, scale=80, base=m)
    m.node_tree.nodes["Principled BSDF"].inputs["Coat Weight"].default_value = 0.7
    return m

NORA = dict(
    name="nora", skin=(0.93, 0.6, 0.45), hair=(0.48, 0.1, 0.02), eyes=(0.15, 0.45, 0.22),
    lip=(0.62, 0.26, 0.25), pants=(0.15, 0.19, 0.3), shoe=(0.5, 0.1, 0.08), ears=False, jaw=0.95,
    lid=(0.3, 0.12), lid_tilt=(3, 8), brows=((0, 0, -1), (4, 6, 5)),
    mouth=[(-9, -26.5), (-2, -27), (6, -25.6), (10, -24)],
    hair_fn=nora_hair, face_extra=nora_face, torso_fn=nora_torso, top=nora_top,
    armL=[(-0.18, 0, 1.17), (-0.29, 0.03, 0.96), (-0.17, -0.03, 0.83)],
    armR=[(0.18, 0, 1.17), (0.24, -0.03, 0.95), (0.16, -0.16, 1.0)], grip=0.55,
)

# ---------------------------------------------------------------- BENNY
def benny_hair(head, m, H):
    def keep(d):
        if d.y < -0.35 and d.z < 0.5: return False
        return d.z > -0.05 or (d.y > 0.4 and d.z > -0.45)
    shell("hair_base", head.c, head.r * 1.05, keep, m, H, thick=0.02)
    random.seed(7)
    for i in range(170):
        yaw = random.uniform(-180, 180); pitch = random.uniform(-5, 44)
        d = head.dirv(yaw, pitch)
        if d.y < -0.55 and pitch < 14: continue
        r = random.uniform(0.02, 0.032)
        ellipsoid("curl", head.point(yaw, pitch, 0.02), (r, r, r * 0.9), m, H, seg=12, sub=1)
    cap = fabric("cap2", (0.72, 0.08, 0.07), rough=0.7, bump_s=0.15, scale=250)
    shell("cap", head.c + Vector((0, 0.02, 0.052)), head.r * 1.17, lambda d: d.z > 0.38, cap, H, thick=0.014)
    ellipsoid("capbrim", head.c + Vector((0, 0.175, 0.1)), (0.09, 0.08, 0.008), cap, H, rot=(math.radians(-22), 0, 0))
    ellipsoid("capbutton", head.c + Vector((0, 0.02, 0.245)), (0.014, 0.014, 0.008), cap, H)

def benny_torso(m, root, w):
    torso_mesh("torso", 0.74, 1.25,
               [(0, 0.2), (0.35, 0.215), (0.65, 0.205), (0.85, 0.215), (0.93, 0.2), (1, 0.06)],
               [(0, 0.14), (0.4, 0.15), (0.7, 0.13), (0.9, 0.12), (1, 0.05)], m, root, belly=0.22)
    hp = mat("headphones", (0.08, 0.08, 0.09), rough=0.35, coat=0.5)
    acc = mat("hp_accent", (1.0, 0.55, 0.05), rough=0.3)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.08, minor_radius=0.01, location=(0, 0.015, 1.235), rotation=(math.radians(-15), 0, 0))
    t = bpy.context.active_object; t.data.materials.append(hp); link(t, root); smooth(t, 1)
    for s in (-1, 1):
        cylinder("cup", (0.075 * s, -0.055, 1.215), 0.032, 0.025, hp, root, rot=(math.radians(70), 0, math.radians(-30 * s)))
        cylinder("cupring", (0.077 * s, -0.067, 1.218), 0.021, 0.009, acc, root, rot=(math.radians(70), 0, math.radians(-30 * s)))

def benny_top():
    m = mat_stripes("shirt_benny2", (0.95, 0.45, 0.1), (0.96, 0.93, 0.85), scale=55)
    fabric("shirt_benny2", None, rough=0.85, bump_s=0.15, base=m)
    return m

BENNY = dict(
    name="benny", skin=(0.58, 0.36, 0.23), hair=(0.05, 0.035, 0.025), eyes=(0.28, 0.15, 0.06),
    lip=(0.42, 0.18, 0.15), pants=(0.5, 0.44, 0.32), shoe=(0.15, 0.42, 0.72), w=1.22, legw=1.15, armw=1.15, handw=1.1,
    lid=(0.7, 0.7), lid_tilt=(0, 0), brows=((5, 7, 5), (5, 7, 5)), jaw=1.08,
    mouth=[(-12, -25), (-5, -27.5), (5, -27.5), (12, -25)],
    hair_fn=benny_hair, torso_fn=benny_torso, top=benny_top,
    armL=[(-0.22, 0, 1.16), (-0.29, 0.02, 0.93), (-0.27, -0.03, 0.72)],
    armR=[(0.22, 0, 1.16), (0.29, 0.02, 0.93), (0.27, -0.03, 0.72)], grip=0.4,
)

# ---------------------------------------------------------------- KAI
def kai_hair(head, m, H):
    def keep(d):
        if d.y < -0.3 and d.z < 0.48: return False
        return d.z > -0.2 or (d.y > 0.3 and d.z > -0.5)
    shell("hair_base", head.c + Vector((0, 0.004, 0.007)), head.r * 1.08, keep, m, H, thick=0.025)
    for yaw, pitch, sx, rz in ((-28, 40, 0.09, 0.5), (-8, 42, 0.085, 0.35), (12, 38, 0.078, 0.25), (30, 30, 0.055, 0.15)):
        q = head.facing(yaw, pitch) @ Quaternion((0, 1, 0), rz)
        ellipsoid("bang", head.point(yaw, pitch, 0.014), (sx, 0.024, 0.042), m, H, rot=q)
    random.seed(3)
    for i in range(16):
        yaw = random.uniform(-70, 70); pitch = random.uniform(55, 88)
        base = head.point(yaw, pitch, 0.012); n = head.normal(yaw, pitch)
        tip = base + n * random.uniform(0.045, 0.075) + Vector((random.uniform(-0.03, 0.03), 0.03, 0))
        tube("tuft", [base, (base + tip) / 2 + n * 0.008, tip], 0.02, m, H, taper=[1.0, 0.7, 0.12])

def kai_face(head, H):
    frame = mat("glasses", (0.06, 0.06, 0.07), rough=0.3, coat=1)
    cs = []
    for s in (-1, 1):
        ec = head.point(22 * s, 3, -0.034 * 0.42); c = ec + Vector((0, -0.05, 0)); cs.append(c)
        bpy.ops.mesh.primitive_torus_add(major_radius=0.045, minor_radius=0.0042, location=c, rotation=(math.radians(90), 0, 0), major_segments=48)
        t = bpy.context.active_object; t.data.materials.append(frame); link(t, H); smooth(t, 1)
        tube("temple", [c + Vector((0.045 * s, 0, 0)), head.point(80 * s, 5, 0.008), head.point(92 * s, 3, 0.0)], 0.0035, frame, H)
    tube("bridge", [cs[0] + Vector((0.044, 0, 0)), (cs[0] + cs[1]) / 2 + Vector((0, 0, 0.008)), cs[1] - Vector((0.044, 0, 0))], 0.004, frame, H)

def kai_torso(m, root, w):
    torso_mesh("torso", 0.74, 1.25,
               [(0, 0.16), (0.35, 0.155), (0.65, 0.17), (0.85, 0.185), (0.93, 0.17), (1, 0.06)],
               [(0, 0.11), (0.5, 0.11), (0.8, 0.115), (1, 0.05)], m, root)
    ellipsoid("hood", (0, 0.07, 1.22), (0.13, 0.07, 0.06), m, root)
    ellipsoid("cuffs", (0, -0.004, 0.77), (0.162, 0.112, 0.035), fabric("rib", (0.08, 0.12, 0.27), scale=500), root)
    s_m = mat("string", (0.95, 0.95, 0.95), rough=0.7)
    for s in (-1, 1):
        tube("string", [(0.035 * s, -0.105, 1.2), (0.04 * s, -0.117, 1.11), (0.043 * s, -0.115, 1.06)], 0.004, s_m, root)
    logo = mat("logo", (0.95, 0.75, 0.2), rough=0.5)
    for r in (0.014, 0.025, 0.036):
        bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=0.003, location=(0.08, -0.108, 1.1), rotation=(math.radians(80), 0, 0))
        t = bpy.context.active_object; t.data.materials.append(logo); link(t, root)

KAI = dict(
    name="kai", skin=(0.9, 0.66, 0.5), hair=(0.03, 0.03, 0.045), eyes=(0.18, 0.11, 0.06),
    lip=(0.55, 0.3, 0.27), pants=(0.22, 0.22, 0.24), shoe=(0.9, 0.9, 0.88),
    lid=(0.42, 0.42), lid_tilt=(-3, -3), brows=((-1, 0, 0), (-1, 0, 0)),
    mouth=[(-7, -25.5), (0, -26.4), (7, -25.5)],
    hair_fn=kai_hair, face_extra=kai_face, torso_fn=kai_torso,
    top=lambda: fabric("hoodie2", (0.1, 0.15, 0.33), scale=400),
    armL=[(-0.18, 0, 1.17), (-0.25, 0.0, 0.95), (-0.2, -0.19, 0.79)],
    armR=[(0.18, 0, 1.17), (0.25, 0.0, 0.95), (0.2, -0.19, 0.79)], grip=0.15,
)
