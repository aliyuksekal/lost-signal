"""Lost Signal - karakter oluşturucu (Blender 5.x, bpy).
Nora, Benny, Kai ve eski radyo 'Old Marlow'.
Kullanım: python3 chars.py [lineup|nora|benny|kai] [hizli]
"""
import bpy, bmesh, math, random, sys
from mathutils import Vector, Quaternion, Euler

# ---------------------------------------------------------------- temel yardımcılar
def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)

_mats = {}
def mat(name, color, rough=0.5, sss=0.0, emis=0.0, alpha=1.0, coat=0.0, metal=0.0):
    if name in _mats:
        return _mats[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if sss:
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
        b.inputs["Subsurface Scale"].default_value = 0.03
    if emis:
        b.inputs["Emission Color"].default_value = (*color, 1)
        b.inputs["Emission Strength"].default_value = emis
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
    if coat:
        b.inputs["Coat Weight"].default_value = coat
    _mats[name] = m
    return m

def mat_stripes(name, c1, c2, scale=9.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.8
    tc = nt.nodes.new("ShaderNodeNewGeometry")  # dünya koordinatı: çizgiler parçalar arasında kesintisiz
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = scale
    sine = nt.nodes.new("ShaderNodeMath"); sine.operation = "SINE"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].color = (*c1, 1)
    ramp.color_ramp.elements[1].position = 0.5
    ramp.color_ramp.elements[1].color = (*c2, 1)
    nt.links.new(tc.outputs["Position"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], mul.inputs[0])
    nt.links.new(mul.outputs[0], sine.inputs[0])
    # sine -1..1 -> ramp 0..1
    add = nt.nodes.new("ShaderNodeMapRange")
    add.inputs["From Min"].default_value = -1
    nt.links.new(sine.outputs[0], add.inputs["Value"])
    nt.links.new(add.outputs[0], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
    return m

def hair_mat(name, color):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = 0.4
    b.inputs["Coat Weight"].default_value = 0.25
    w = nt.nodes.new("ShaderNodeTexWave")
    w.wave_type = "BANDS"; w.bands_direction = "X"
    w.inputs["Scale"].default_value = 18; w.inputs["Distortion"].default_value = 4
    w.inputs["Detail"].default_value = 3
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Object"], w.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.35
    nt.links.new(w.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m

def link(obj, parent):
    obj.parent = parent
    return obj

def smooth(obj, level=2):
    for p in obj.data.polygons:
        p.use_smooth = True
    if level:
        mod = obj.modifiers.new("sub", "SUBSURF")
        mod.levels = level; mod.render_levels = level
    return obj

def new_obj(name, bm, material, parent):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(material)
    return link(ob, parent)

def ellipsoid(name, loc, radii, material, parent, rot=None, seg=32, sub=1):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=1.0)
    ob = new_obj(name, bm, material, parent)
    ob.location = loc
    ob.scale = radii
    if rot is not None:
        if isinstance(rot, Quaternion):
            ob.rotation_mode = "QUATERNION"; ob.rotation_quaternion = rot
        else:
            ob.rotation_euler = rot
    return smooth(ob, sub)

def shell(name, center, radii, keep, material, parent, thick=0.012, seg=64):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=1.0)
    dele = [v for v in bm.verts if not keep(v.co.normalized())]
    bmesh.ops.delete(bm, geom=dele, context="VERTS")
    ob = new_obj(name, bm, material, parent)
    ob.location = center; ob.scale = radii
    s = ob.modifiers.new("solid", "SOLIDIFY"); s.thickness = thick / max(radii); s.offset = 0
    return smooth(ob, 2)

def tube(name, pts, radius, material, parent, taper=None):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius; cu.bevel_resolution = 6; cu.use_fill_caps = True
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts) - 1)
    for bp, p in zip(sp.bezier_points, pts):
        bp.co = p; bp.handle_left_type = bp.handle_right_type = "AUTO"
        if taper:
            bp.radius = taper.pop(0)
    ob = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(material)
    return link(ob, parent)

def rounded_box(name, loc, size, material, parent, bevel=0.02, rot=(0, 0, 0)):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    ob = new_obj(name, bm, material, parent)
    ob.location = loc; ob.scale = size; ob.rotation_euler = rot
    bv = ob.modifiers.new("bev", "BEVEL"); bv.width = bevel; bv.segments = 5
    bv.affect = "EDGES"
    ob.modifiers["bev"].use_clamp_overlap = True
    for p in ob.data.polygons: p.use_smooth = True
    # scale before bevel: apply scale so bevel stays round
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ob.select_set(False)
    return ob

def cylinder(name, loc, r, depth, material, parent, rot=(0, 0, 0), seg=32):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r, depth=depth)
    ob = new_obj(name, bm, material, parent)
    ob.location = loc; ob.rotation_euler = rot
    bv = ob.modifiers.new("bev", "BEVEL"); bv.width = min(r, depth) * 0.25; bv.segments = 3
    for p in ob.data.polygons: p.use_smooth = True
    return ob

# ---------------------------------------------------------------- kafa geometrisi
class Head:
    def __init__(self, c, r):
        self.c = Vector(c); self.r = Vector(r)

    def dirv(self, yaw, pitch):
        y, p = math.radians(yaw), math.radians(pitch)
        return Vector((math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p)))

    def point(self, yaw, pitch, out=0.0):
        d = self.dirv(yaw, pitch)
        p = self.c + Vector((d.x * self.r.x, d.y * self.r.y, d.z * self.r.z))
        return p + self.normal(yaw, pitch) * out

    def normal(self, yaw, pitch):
        d = self.dirv(yaw, pitch)
        return Vector((d.x / self.r.x, d.y / self.r.y, d.z / self.r.z)).normalized()

    def facing(self, yaw, pitch):
        return self.normal(yaw, pitch).to_track_quat("-Y", "Z")

def face_curve(name, head, yp_list, out, radius, material, parent, taper=None):
    pts = [head.point(y, p, out) for y, p in yp_list]
    return tube(name, pts, radius, material, parent, taper)

# ---------------------------------------------------------------- karakter parçaları
def build_eye(tag, head, yaw, pitch, iris_col, lid, lid_tilt, skin, parent, look=Vector((0, -6, 1.0)), size=0.058):
    n = head.normal(yaw, pitch)
    ec = head.point(yaw, pitch, -size * 0.45)
    white = mat("sclera", (0.95, 0.94, 0.92), rough=0.15, coat=1.0)
    ellipsoid(tag + "_eye", ec, (size, size, size), white, parent, sub=1)
    d = (look - ec).normalized()
    q = d.to_track_quat("-Y", "Z")
    irm = mat("iris_" + tag, iris_col, rough=0.25, coat=1.0)
    ellipsoid(tag + "_iris", ec + d * (size - 0.006), (size * 0.62, 0.012, size * 0.62), irm, parent, rot=q)
    ellipsoid(tag + "_pupil", ec + d * (size + 0.001), (size * 0.32, 0.008, size * 0.32), mat("pupil", (0.01, 0.01, 0.012), rough=0.2, coat=1), parent, rot=q)
    # ışık parıltısı
    right = d.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(d).normalized()
    hl = ec + d * (size + 0.006) + up * size * 0.3 - right * size * 0.25
    ellipsoid(tag + "_hl", hl, (size * 0.13, 0.004, size * 0.13), mat("catch", (1, 1, 1), emis=6.0), parent, rot=q)
    # göz kapağı (üst)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
    dele = [v for v in bm.verts if v.co.z < lid]
    bmesh.ops.delete(bm, geom=dele, context="VERTS")
    ob = new_obj(tag + "_lid", bm, skin, parent)
    ob.location = ec; ob.scale = (size * 1.07,) * 3
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = n.to_track_quat("-Y", "Z") @ Quaternion((0, 1, 0), math.radians(lid_tilt))
    s = ob.modifiers.new("solid", "SOLIDIFY"); s.thickness = 0.12
    smooth(ob, 2)
    # kirpik çizgisi
    return ec

def build_body(P, root):
    """P: parametre sözlüğü. Karakter yerel uzayda (0,0) üzerinde, yüzü -Y."""
    w = P.get("w", 1.0)
    skin = mat("skin_" + P["name"], P["skin"], rough=0.45, sss=0.25)
    hairm = hair_mat("hair_" + P["name"], P["hair"])
    pants = mat("pants_" + P["name"], P["pants"], rough=0.85)
    shoe = mat("shoe_" + P["name"], P["shoe"], rough=0.5)
    sole = mat("sole", (0.92, 0.9, 0.86), rough=0.6)
    top = P["top_mat"]
    hz = 1.30
    head = Head((0, 0, hz), (0.215, 0.205, 0.235))

    # --- kafa
    ellipsoid("head", head.c, head.r, skin, root, sub=2)
    # çene/yanaklar dolgunluğu
    ellipsoid("jaw", (0, -0.02, hz - 0.08), (0.17, 0.16, 0.14), skin, root, sub=2)
    ellipsoid("neck", (0, 0.01, 1.04), (0.055, 0.05, 0.09), skin, root)
    # burun
    ellipsoid("nose", head.point(0, -8, 0.0), P.get("nose", (0.03, 0.03, 0.028)), skin, root, rot=head.facing(0, -8))
    # kulaklar
    if P.get("ears", True):
        for s in (-1, 1):
            ellipsoid("ear", head.point(88 * s, -4, -0.01), (0.03, 0.045, 0.055), skin, root, rot=head.facing(88 * s, -4))
    # gözler
    for s, tilt in ((-1, -P["lid_tilt"][0]), (1, P["lid_tilt"][1])):
        build_eye(P["name"] + "_eye%d" % s, head, 23 * s, 2, P["eyes"], P["lid"][0 if s < 0 else 1], tilt * s, skin, root)
    # kaşlar
    bm_ = mat("brow_" + P["name"], tuple(c * 0.7 for c in P["hair"]), rough=0.6)
    for s in (-1, 1):
        raise_ = P["brows"][0 if s < 0 else 1]
        pts = [(12 * s, 18 + raise_[0]), (23 * s, 21 + raise_[1]), (34 * s, 18 + raise_[2])]
        face_curve("brow", head, pts, 0.008, 0.012, bm_, root)
    # ağız
    P["mouth"](head, root)
    # yanak kızarıklığı
    blush = mat("blush", (0.95, 0.45, 0.45), rough=0.6, alpha=0.35)
    for s in (-1, 1):
        ellipsoid("blush", head.point(36 * s, -14, 0.002), (0.04, 0.004, 0.025), blush, root, rot=head.facing(36 * s, -14))
    # saç
    P["hair_fn"](head, hairm, root)

    # --- gövde
    P["torso_fn"](top, root, w)
    ellipsoid("hips", (0, 0.0, 0.56), (0.15 * w, 0.11 * w, 0.09), pants, root)
    # bacaklar ve ayakkabılar
    for s in (-1, 1):
        tube("leg", [(0.075 * s * w, 0, 0.58), (0.085 * s * w, -0.01, 0.33), (0.09 * s * w, 0, 0.1)], 0.065 * P.get("legw", 1.0), pants, root)
        ellipsoid("shoe", (0.09 * s * w, -0.05, 0.055), (0.068, 0.125, 0.052), shoe, root, sub=2)
        ellipsoid("sole", (0.09 * s * w, -0.05, 0.02), (0.075, 0.133, 0.022), sole, root, sub=2)
    # kollar
    sleeve = P.get("sleeve_mat", top)
    for side, pts in (("L", P["armL"]), ("R", P["armR"])):
        tube("arm" + side, [Vector(p) for p in pts], 0.045 * P.get("armw", 1.0), sleeve, root)
        hp = Vector(pts[-1])
        hand_dir = (hp - Vector(pts[-2])).normalized()
        ellipsoid("hand" + side, hp + hand_dir * 0.04, (0.042, 0.04, 0.05), skin, root, rot=hand_dir.to_track_quat("Z", "Y"))
        ellipsoid("thumb" + side, hp + hand_dir * 0.03 + Vector((0, -0.035, 0)), (0.016, 0.016, 0.026), skin, root)
    if "extra" in P:
        P["extra"](head, root)
    return head

# ---------------------------------------------------------------- NORA
def nora_hair(head, m, root):
    def keep(d):
        if d.y < -0.22 and d.z < 0.2:  # yüz açıklığı (küt kakül)
            return False
        return d.z > -0.62
    shell("hair_bob", head.c + Vector((0, 0.012, 0.0)), head.r * 1.09, keep, m, root, thick=0.03)
    # bob alt uçları içe kıvrık
    for s in (-1, 1):
        ellipsoid("bob_tip", head.c + Vector((0.2 * s, 0.02, -0.17)), (0.06, 0.13, 0.05), m, root, rot=(0, math.radians(25 * s), 0))
    # kakül tutamları
    for yaw in range(-42, 43, 12):
        top = head.point(yaw * 0.6, 55, 0.035)
        mid = head.point(yaw * 0.9, 34, 0.03)
        end = head.point(yaw + 4, 16, 0.022)
        tube("fringe", [top, mid, end], 0.032, m, root, taper=[1.0, 0.9, 0.45])
    ellipsoid("bob_back", head.c + Vector((0, 0.1, -0.15)), (0.2, 0.1, 0.07), m, root)

def nora_mouth(head, root):
    lip = mat("lip_nora", (0.55, 0.16, 0.16), rough=0.5)
    # yamuk sırıtma (kuşkucu)
    face_curve("mouth", head, [(-11, -27), (-2, -27.5), (8, -25.5), (13, -23)], 0.004, 0.0085, lip, root)

def nora_torso(m, root, w):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
    for v in bm.verts:
        f = 1.0 + max(0.0, -v.co.z) * 0.45  # alta doğru genişleyen A-kesim
        v.co.x *= f; v.co.y *= f
    ob = new_obj("raincoat", bm, m, root)
    ob.location = (0, 0, 0.75); ob.scale = (0.17, 0.125, 0.27)
    smooth(ob, 2)
    ellipsoid("shoulders", (0, 0, 0.94), (0.17, 0.12, 0.07), m, root)
    collar = mat("coat_dark", (0.62, 0.38, 0.05), rough=0.4, coat=0.6)
    for s in (-1, 1):
        ellipsoid("collar", (0.075 * s, -0.08, 0.99), (0.07, 0.03, 0.045), collar, root, rot=(0.3, 0, -0.5 * s))
    btn = mat("button", (0.25, 0.14, 0.06), rough=0.3, coat=1)
    for i, z in enumerate((0.92, 0.8, 0.68, 0.56)):
        ellipsoid("btn", (0.0, -0.205 + (0.92 - z) * -0.09, z), (0.017, 0.008, 0.017), btn, root)
    # cep
    for s in (-1, 1):
        ellipsoid("pocketflap", (0.13 * s, -0.17, 0.62), (0.06, 0.015, 0.018), collar, root, rot=(0, 0, 0.35 * s))

def nora_extra(head, root):
    fr = mat("freckle", (0.55, 0.25, 0.12), rough=0.6)
    random.seed(4)
    for s in (-1, 1):
        for i in range(7):
            y = 30 + random.uniform(-9, 9); p = -9 + random.uniform(-5, 5)
            ellipsoid("freckle", head.point(y * s, p, 0.0015), (0.006, 0.002, 0.006), fr, root, rot=head.facing(y * s, p), seg=8, sub=0)
    # defter (sağ elde)
    book = mat("notebook", (0.7, 0.08, 0.08), rough=0.5)
    rounded_box("notebook", (0.15, -0.23, 0.84), (0.11, 0.025, 0.15), book, root, bevel=0.006, rot=(0.25, 0, -0.25))
    pages = mat("pages", (0.95, 0.93, 0.85), rough=0.8)
    rounded_box("pages", (0.152, -0.222, 0.845), (0.105, 0.018, 0.143), pages, root, bevel=0.003, rot=(0.25, 0, -0.25))
    # kalem kulakta
    tube("pencil", [head.point(80, 14, 0.03), head.point(70, 24, 0.07)], 0.007, mat("pencil", (0.95, 0.7, 0.1)), root)

NORA = dict(
    name="nora", skin=(0.95, 0.6, 0.44), hair=(0.5, 0.1, 0.02), eyes=(0.18, 0.5, 0.25),
    pants=(0.16, 0.2, 0.32), shoe=(0.55, 0.12, 0.1), ears=False,
    lid=(0.28, 0.05), lid_tilt=(4, 10),  # bir göz daha kısık = kuşku
    brows=((0, 0, -1), (5, 7, 6)),  # sağ kaş kalkık
    mouth=nora_mouth, hair_fn=nora_hair, torso_fn=nora_torso, extra=nora_extra,
    armL=[(-0.17, 0, 0.94), (-0.28, 0.03, 0.76), (-0.17, -0.02, 0.62)],  # el belde
    armR=[(0.17, 0, 0.94), (0.24, -0.04, 0.75), (0.16, -0.17, 0.8)],
)

# ---------------------------------------------------------------- BENNY
def benny_hair(head, m, root):
    def keep(d):
        if d.y < -0.35 and d.z < 0.5:
            return False
        return d.z > -0.05 or (d.y > 0.4 and d.z > -0.45)
    shell("hair_base", head.c, head.r * 1.06, keep, m, root, thick=0.03)
    random.seed(7)
    for i in range(120):
        yaw = random.uniform(-180, 180); pitch = random.uniform(-5, 42)
        d = head.dirv(yaw, pitch)
        if d.y < -0.55 and pitch < 12:
            continue
        r = random.uniform(0.032, 0.048)
        ellipsoid("curl", head.point(yaw, pitch, 0.03), (r, r, r), m, root, seg=12, sub=1)
    # ters şapka
    cap = mat("cap", (0.78, 0.1, 0.08), rough=0.55)
    shell("cap", head.c + Vector((0, 0.03, 0.075)), head.r * 1.17, lambda d: d.z > 0.38, cap, root, thick=0.02)
    ellipsoid("capbrim", head.c + Vector((0, 0.25, 0.14)), (0.13, 0.11, 0.012), cap, root, rot=(math.radians(-22), 0, 0))
    ellipsoid("capbutton", head.c + Vector((0, 0.03, 0.35)), (0.02, 0.02, 0.012), cap, root)

def benny_mouth(head, root):
    q = head.facing(0, -27)
    ellipsoid("mouth", head.point(0, -27, -0.006), (0.072, 0.012, 0.042), mat("mouthin", (0.3, 0.04, 0.06), rough=0.5), root, rot=q)
    ellipsoid("teeth", head.point(0, -22.5, -0.001), (0.055, 0.008, 0.014), mat("teeth", (0.97, 0.96, 0.92), rough=0.3), root, rot=q)
    ellipsoid("tongue", head.point(0, -31, -0.002), (0.035, 0.008, 0.016), mat("tongue", (0.85, 0.3, 0.32), rough=0.4), root, rot=q)

def benny_torso(m, root, w):
    ellipsoid("torso", (0, 0, 0.77), (0.205, 0.165, 0.25), m, root, sub=2)
    ellipsoid("belly", (0, -0.03, 0.7), (0.19, 0.16, 0.17), m, root, sub=2)
    # kulaklık boyunda
    hp = mat("headphones", (0.08, 0.08, 0.09), rough=0.35, coat=0.5)
    accent = mat("hp_accent", (1.0, 0.55, 0.05), rough=0.3)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.11, minor_radius=0.014, location=(0, 0.02, 1.0), rotation=(math.radians(-15), 0, 0))
    t = bpy.context.active_object; t.data.materials.append(hp); link(t, root); smooth(t, 1)
    for s in (-1, 1):
        c = cylinder("cup", (0.1 * s, -0.07, 0.97), 0.045, 0.035, hp, root, rot=(math.radians(70), 0, math.radians(-30 * s)))
        cylinder("cupring", (0.103 * s, -0.087, 0.975), 0.03, 0.012, accent, root, rot=(math.radians(70), 0, math.radians(-30 * s)))

def benny_extra(head, root):
    pass

BENNY = dict(
    name="benny", skin=(0.62, 0.4, 0.26), hair=(0.06, 0.04, 0.03), eyes=(0.3, 0.17, 0.07),
    pants=(0.55, 0.48, 0.35), shoe=(0.15, 0.45, 0.75), w=1.25, legw=1.2, armw=1.18,
    lid=(0.8, 0.8), lid_tilt=(0, 0),
    brows=((6, 9, 6), (6, 9, 6)), nose=(0.04, 0.035, 0.034),
    mouth=benny_mouth, hair_fn=benny_hair, torso_fn=benny_torso, extra=benny_extra,
    armL=[(-0.22, 0, 0.93), (-0.3, 0.0, 0.74), (-0.29, -0.04, 0.56)],
    armR=[(0.22, 0, 0.93), (0.36, -0.02, 1.08), (0.37, -0.06, 1.3)],  # el sallıyor
)

# ---------------------------------------------------------------- KAI
def kai_hair(head, m, root):
    def keep(d):
        if d.y < -0.3 and d.z < 0.48:
            return False
        return d.z > -0.2 or (d.y > 0.3 and d.z > -0.5)
    shell("hair_base", head.c + Vector((0, 0.005, 0.01)), head.r * 1.09, keep, m, root, thick=0.035)
    # yana savrulan perçem
    for i, (yaw, pitch, sx, rz) in enumerate(((-28, 40, 0.13, 0.5), (-8, 42, 0.12, 0.35), (12, 38, 0.11, 0.25), (30, 30, 0.08, 0.15))):
        q = head.facing(yaw, pitch) @ Quaternion((0, 1, 0), rz)
        ellipsoid("bang", head.point(yaw, pitch, 0.02), (sx, 0.035, 0.06), m, root, rot=q)
    # tepede dikleşen tutamlar
    for yaw, pitch, tilt in ((-15, 70, -0.5), (15, 72, 0.3), (40, 62, 0.8), (-45, 60, -0.9), (0, 85, 0.0)):
        base = head.point(yaw, pitch, 0.02)
        n = head.normal(yaw, pitch)
        tip = base + n * 0.1 + Vector((tilt * 0.04, 0.04, 0))
        tube("tuft", [base, (base + tip) / 2 + n * 0.01, tip], 0.03, m, root, taper=[1.0, 0.7, 0.15])

def kai_mouth(head, root):
    lip = mat("lip_kai", (0.45, 0.2, 0.18), rough=0.5)
    face_curve("mouth", head, [(-8, -25), (0, -26.2), (8, -25)], 0.004, 0.0075, lip, root)

def kai_torso(m, root, w):
    ellipsoid("torso", (0, 0, 0.77), (0.17, 0.12, 0.25), m, root, sub=2)
    ellipsoid("hood", (0, 0.08, 1.0), (0.15, 0.08, 0.075), m, root)
    ellipsoid("cuffs", (0, -0.005, 0.56), (0.165, 0.115, 0.04), mat("hoodie_rib", (0.1, 0.14, 0.3), rough=0.9), root)
    s_m = mat("string", (0.95, 0.95, 0.95), rough=0.7)
    for s in (-1, 1):
        tube("string", [(0.04 * s, -0.11, 0.97), (0.045 * s, -0.125, 0.88), (0.05 * s, -0.12, 0.84)], 0.006, s_m, root)
    # göğüs logosu: küçük sinyal halkaları
    logo = mat("logo", (0.95, 0.75, 0.2), rough=0.5)
    for i, r in enumerate((0.02, 0.035, 0.05)):
        bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=0.004, location=(0.09, -0.112, 0.9 - i * 0.0), rotation=(math.radians(80), 0, 0))
        t = bpy.context.active_object; t.data.materials.append(logo); link(t, root)

def kai_extra(head, root):
    frame = mat("glasses", (0.06, 0.06, 0.07), rough=0.3, coat=1)
    glass = mat("lens", (0.9, 0.95, 1.0), rough=0.02, alpha=0.08)
    centers = []
    for s in (-1, 1):
        ec = head.point(23 * s, 2, -0.058 * 0.45)
        c = ec + Vector((0, -0.075, 0))
        centers.append(c)
        bpy.ops.mesh.primitive_torus_add(major_radius=0.068, minor_radius=0.0065, location=c, rotation=(math.radians(90), 0, 0), major_segments=48)
        t = bpy.context.active_object; t.data.materials.append(frame); link(t, root); smooth(t, 1)
        # sap
        tube("temple", [c + Vector((0.068 * s, 0, 0)), head.point(80 * s, 4, 0.01), head.point(92 * s, 2, 0.0)], 0.005, frame, root)
    tube("bridge", [centers[0] + Vector((0.066, 0, 0)), (centers[0] + centers[1]) / 2 + Vector((0, 0, 0.012)), centers[1] - Vector((0.066, 0, 0))], 0.006, frame, root)
    build_radio(root, (0, -0.25, 0.8), 1.0)

KAI = dict(
    name="kai", skin=(0.93, 0.68, 0.5), hair=(0.035, 0.035, 0.05), eyes=(0.2, 0.13, 0.08),
    pants=(0.25, 0.25, 0.27), shoe=(0.92, 0.92, 0.9),
    lid=(0.42, 0.42), lid_tilt=(-3, -3),
    brows=((-1, 0, 0), (-1, 0, 0)),
    mouth=kai_mouth, hair_fn=kai_hair, torso_fn=kai_torso, extra=kai_extra,
    armL=[(-0.17, 0, 0.94), (-0.23, -0.04, 0.74), (-0.15, -0.22, 0.78)],
    armR=[(0.17, 0, 0.94), (0.23, -0.04, 0.74), (0.15, -0.22, 0.78)],
)

# ---------------------------------------------------------------- RADYO "Old Marlow"
def build_radio(root, loc, sc):
    loc = Vector(loc)
    wood = mat("wood", (0.33, 0.14, 0.05), rough=0.35, coat=0.8)
    rounded_box("radio_body", loc, (0.24 * sc, 0.12 * sc, 0.16 * sc), wood, root, bevel=0.025)
    cloth = mat("grille", (0.75, 0.62, 0.42), rough=0.95)
    cylinder("speaker", loc + Vector((-0.05, -0.061, 0.0)), 0.05, 0.01, cloth, root, rot=(math.radians(90), 0, 0))
    gold = mat("brass", (0.85, 0.6, 0.25), rough=0.25, metal=1.0)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.052, minor_radius=0.005, location=loc + Vector((-0.05, -0.066, 0)), rotation=(math.radians(90), 0, 0))
    t = bpy.context.active_object; t.data.materials.append(gold); link(t, root)
    dial = mat("dial", (1.0, 0.62, 0.18), emis=4.0)
    rounded_box("dial", loc + Vector((0.055, -0.062, 0.025)), (0.07, 0.006, 0.035), dial, root, bevel=0.004)
    needle = mat("needle", (0.6, 0.05, 0.02), rough=0.4)
    rounded_box("needle", loc + Vector((0.065, -0.067, 0.025)), (0.003, 0.003, 0.032), needle, root, bevel=0.001)
    for dx in (0.035, 0.075):
        cylinder("knob", loc + Vector((dx, -0.065, -0.035)), 0.013, 0.014, gold, root, rot=(math.radians(90), 0, 0))
    tube("antenna", [loc + Vector((0.09, 0.02, 0.08)), loc + Vector((0.2, 0.05, 0.16)), loc + Vector((0.32, 0.08, 0.25))], 0.0035, mat("chrome", (0.9, 0.9, 0.92), rough=0.15, metal=1.0), root)
    ellipsoid("antenna_tip", loc + Vector((0.32, 0.08, 0.25)), (0.009,) * 3, gold, root)
    # sinyal parıltısı
    pt = bpy.data.lights.new("dial_glow", "POINT"); pt.energy = 3; pt.color = (1, 0.6, 0.2); pt.shadow_soft_size = 0.02
    o = bpy.data.objects.new("dial_glow", pt); o.location = loc + Vector((0.055, -0.09, 0.025))
    bpy.context.collection.objects.link(o); link(o, root)

# ---------------------------------------------------------------- sahne
def character(P, loc, scale=1.0, yaw=0.0):
    root = bpy.data.objects.new(P["name"], None)
    bpy.context.collection.objects.link(root)
    if P["name"] == "benny":
        P["top_mat"] = mat_stripes("shirt_benny", (0.98, 0.5, 0.12), (0.97, 0.94, 0.86), scale=40)
        P["sleeve_mat"] = P["top_mat"]
    elif P["name"] == "nora":
        P["top_mat"] = mat("raincoat", (0.95, 0.6, 0.02), rough=0.3, coat=0.7)
    else:
        P["top_mat"] = mat("hoodie", (0.12, 0.18, 0.38), rough=0.9)
    build_body(P, root)
    root.location = loc; root.scale = (scale,) * 3; root.rotation_euler = (0, 0, math.radians(yaw))
    return root

def stage(cam_loc, cam_target, lens=45, bg=(0.1, 0.17, 0.3)):
    # kıvrık fon (cyclorama)
    verts, faces = [], []
    prof = []
    for i in range(0, 21):
        a = math.radians(90 * i / 20)
        prof.append((0, 3 - 1.2 * math.sin(a) + 0.0, 1.2 - 1.2 * math.cos(a)))
    prof = [(0, -6, 0)] + [(0, 1.8, 0)] + [(0, 1.8 + (3 - 1.8) * 0 + 1.2 * math.sin(math.radians(90 * i / 20)), 1.2 - 1.2 * math.cos(math.radians(90 * i / 20))) for i in range(1, 21)] + [(0, 3.0, 6)]
    W = 12
    for x in (-W, W):
        for p in prof:
            verts.append((x, p[1], p[2]))
    n = len(prof)
    for i in range(n - 1):
        faces.append((i, i + 1, n + i + 1, n + i))
    me = bpy.data.meshes.new("cyc"); me.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new("cyc", me); bpy.context.collection.objects.link(ob)
    for p in me.polygons: p.use_smooth = True
    ob.data.materials.append(mat("backdrop", bg, rough=0.9))
    # ışıklar
    def area(name, loc, target, energy, color, size):
        L = bpy.data.lights.new(name, "AREA"); L.energy = energy; L.color = color; L.size = size
        o = bpy.data.objects.new(name, L); o.location = loc
        bpy.context.collection.objects.link(o)
        o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    area("key", (-2.5, -3.2, 3.2), (0, 0, 1.0), 380, (1.0, 0.9, 0.8), 2.5)
    area("fill", (3.0, -2.5, 1.6), (0, 0, 1.0), 110, (0.75, 0.85, 1.0), 3.0)
    area("rim", (0.5, 2.2, 3.0), (0, 0, 1.1), 450, (1.0, 0.85, 0.7), 1.5)
    area("top", (0, 0, 4.0), (0, 0, 0), 50, (1, 1, 1), 4)
    w = bpy.data.worlds.new("w"); bpy.context.scene.world = w
    w.use_nodes = True; w.node_tree.nodes["Background"].inputs[0].default_value = (0.25, 0.3, 0.4, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.2
    cam = bpy.data.cameras.new("cam"); cam.lens = lens
    cam.dof.use_dof = True; cam.dof.aperture_fstop = 4.0
    co = bpy.data.objects.new("cam", cam); bpy.context.collection.objects.link(co)
    co.location = cam_loc
    co.rotation_euler = (Vector(cam_target) - Vector(cam_loc)).to_track_quat("-Z", "Y").to_euler()
    cam.dof.focus_distance = (Vector(cam_target) - Vector(cam_loc)).length
    bpy.context.scene.camera = co

def render(path, res=(1600, 900), samples=96):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Punchy"
    except Exception:
        pass
    sc.render.filepath = path
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "lineup"
    fast = "hizli" in sys.argv
    clear()
    if what == "lineup":
        character(NORA, (-0.78, 0.05, 0), 1.0, 12)
        character(KAI, (0, 0, 0), 1.07, 0)
        character(BENNY, (0.8, 0.05, 0), 0.97, -12)
        stage((0, -4.3, 1.05), (0, 0, 0.88), lens=45)
    else:
        P = {"nora": NORA, "benny": BENNY, "kai": KAI}[what]
        character(P, (0, 0, 0), 1.0, 0)
        stage((0.0, -3.4, 0.95), (0, 0, 0.78), lens=45)
    out = "/home/claude/lostsignal/renders/%s%s.png" % (what, "_test" if fast else "")
    render(out, (800, 450) if fast else (1600, 900), 24 if fast else 128)
    bpy.ops.wm.save_as_mainfile(filepath="/home/claude/lostsignal/%s.blend" % what)
    print("OK", out)
