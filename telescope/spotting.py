"""观鸟镜（折射式，看地面景物）：物镜 + 镜筒 + 调焦座 + 45° 正像棱镜 + 目镜。

物镜外协，正像棱镜、目镜、调焦座买成品，镜筒和其余结构件 CNC / 3D 打印。
沿光轴的位置都从物镜后表面量起，单位 mm。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import stl
from .optics import EYE_PUPIL_MM, Telescope, cone_overlap

FIELD_STOP_1_25IN_MM = 27.0  # 1.25 寸目镜镜筒能容纳的最大视场光阑
TRANSMISSION = 0.85  # 物镜、棱镜、目镜全部多层镀膜后的总透过率（估计）
PRACTICAL_MAX_X = 60  # 看地面景物时地面热气流的限制，再高也不会更清楚
ALUMINIUM_KG_MM3 = 2.7e-6
PRISMS = {  # (光程, 通光孔径, 入光面到焦面)
    "2": (110.0, 45.0, 70.0),
    "1.25": (95.0, 25.0, 50.0),
}


@dataclass(frozen=True)
class EyepieceOption:
    name: str
    focal_mm: float
    afov_deg: float  # 目镜视场角

    @property
    def field_stop_mm(self) -> float:
        return min(2 * self.focal_mm * math.tan(math.radians(self.afov_deg / 2)), FIELD_STOP_1_25IN_MM)


# 推荐的目镜组合：傍晚用 40 mm，平时 32 mm 找目标，变焦目镜看细节
EYEPIECES = (
    EyepieceOption("40 mm 普洛（傍晚、月光下）", 40, 43),
    EyepieceOption("32 mm 普洛（找目标）", 32, 52),
    EyepieceOption("8–24 mm 变焦（24 mm 端）", 24, 40),
    EyepieceOption("8–24 mm 变焦（8 mm 端）", 8, 60),
)


@dataclass(frozen=True)
class SpottingSpec:
    aperture_mm: float = 80.0
    focal_length_mm: float = 480.0
    ed_glass: bool = True  # 物镜用 ED 低色散玻璃
    tube_od_mm: float = 90.0  # 6061 铝管，外径 90，壁厚 2
    tube_wall_mm: float = 2.0
    cell_rear_mm: float = 10.0  # 物镜后表面到镜座后端面（镜筒从这里开始）
    focuser_length_mm: float = 70.0  # 调焦座收到底时，从镜筒后端到抽筒顶端
    focuser_travel_mm: float = 50.0
    focuser_bore_mm: float = 48.0  # 2 寸调焦座的通光孔径
    # 2 寸 45° 正像棱镜。1.25 寸的便宜，但通光孔径只有 25 mm 左右，低倍目镜视场边缘会暗一半
    prism_path_mm: float = 110.0  # 正像棱镜光程：从抽筒顶端到目镜台阶（含 2 寸转 1.25 寸转接环）
    prism_clear_mm: float = 45.0  # 棱镜通光孔径
    prism_to_focus_mm: float = 70.0  # 棱镜入光面到目镜焦面的光程
    focus_margin_mm: float = 5.0  # 看无穷远时抽筒至少伸出这么多（不同目镜焦面位置差几毫米）
    baffles: int = 3

    def __post_init__(self):
        if self.focal_length_mm < 4 * self.aperture_mm:
            raise ValueError("焦比小于 f/4 的折射物镜色差太大，不适合自己做")
        if self.tube_od_mm - 2 * self.tube_wall_mm < self.aperture_mm + 4:
            raise ValueError("镜筒内径至少要比物镜大 4 mm")


@dataclass
class EyepieceRow:
    eyepiece: EyepieceOption
    magnification: float
    exit_pupil_mm: float
    true_fov_deg: float
    width_at_100m: float
    twilight_factor: float
    brightness_adult: float  # 和肉眼看同一处景物相比的亮度（成人傍晚瞳孔约 6 mm）
    brightness_child: float  # 孩子傍晚瞳孔约 7 mm
    edge_light: float  # 视场边缘的亮度（中心为 1），被棱镜和调焦座挡掉的部分
    notes: list[str] = field(default_factory=list)


@dataclass
class Baffle:
    position_mm: float  # 离镜筒前端（镜座后端面）
    hole_mm: float


@dataclass
class SpottingDesign:
    spec: SpottingSpec
    rows: list[EyepieceRow]
    min_focus_m: float
    tube_length_mm: float
    dew_shield_mm: float
    baffles: list[Baffle]
    ca_index: float
    color_note: str
    balance_from_cell_mm: float  # 三脚架夹座中心离镜筒前端
    total_kg: float
    warnings: list[str] = field(default_factory=list)

    @property
    def telescope(self) -> Telescope:
        return Telescope(self.spec.aperture_mm, self.spec.focal_length_mm)


LINER_WALL_MM = 1.2  # 打印的内衬筒壁厚
LINER_FIT_MM = 0.2  # 内衬筒和镜筒之间单边留的间隙


def light_cone_mm(aperture_mm: float, focal_mm: float, x_mm: float) -> float:
    """离物镜 x_mm 处，要让 1.25 寸目镜整个视场都全亮，光锥需要多粗。"""
    fs = FIELD_STOP_1_25IN_MM
    return fs + (aperture_mm - fs) * (focal_mm - x_mm) / focal_mm


def focus_extension_mm(focal_mm: float, distance_m: float) -> float:
    """看 distance_m 远的东西，焦点要比看无穷远时往后挪多少（薄透镜公式）。"""
    u = distance_m * 1000
    return focal_mm * u / (u - focal_mm) - focal_mm


def nearest_focus_m(focal_mm: float, extension_mm: float) -> float:
    if extension_mm <= 0:
        return math.inf
    return focal_mm * (focal_mm + extension_mm) / extension_mm / 1000


def twilight_factor(aperture_mm: float, magnification: float) -> float:
    """“黄昏系数” √(口径 × 倍率)：傍晚能看出多少细节的经验指标，8×42 约 18，越大越好。"""
    return math.sqrt(aperture_mm * magnification)


def relative_brightness(exit_pupil_mm: float, eye_pupil_mm: float) -> float:
    """看大面积景物时，和肉眼直接看相比有多亮。出瞳比瞳孔小时，进眼睛的光变少。"""
    return TRANSMISSION * min(1.0, (exit_pupil_mm / eye_pupil_mm) ** 2)


def chromatic_index(aperture_mm: float, focal_mm: float) -> float:
    """色差指数：焦比 ÷ 口径（英寸）。普通消色差物镜低于 3 时高倍有明显紫边。"""
    return (focal_mm / aperture_mm) / (aperture_mm / 25.4)


def design_spotting(spec: SpottingSpec = SpottingSpec()) -> SpottingDesign:
    d, f = spec.aperture_mm, spec.focal_length_mm
    warnings = []

    # 镜筒长度：物镜焦距 = 镜座 + 镜筒 + 调焦座 + 余量 + 棱镜光程
    tube = f - spec.cell_rear_mm - spec.focuser_length_mm - spec.focus_margin_mm - spec.prism_path_mm
    if tube < 2 * d:
        warnings.append("镜筒太短，调焦座和正像棱镜占了大部分焦距，换焦距更长的物镜")
    min_focus = nearest_focus_m(f, spec.focuser_travel_mm - spec.focus_margin_mm)

    # 会挡光的圆孔：(名字, 孔径, 离焦面多远)
    stops = [
        ("调焦座", spec.focuser_bore_mm, spec.focuser_length_mm + spec.focus_margin_mm + spec.prism_path_mm),
        ("正像棱镜", spec.prism_clear_mm, spec.prism_to_focus_mm),
    ]

    def edge_light(field_radius: float) -> float:
        return min(cone_overlap(d / 2 * s / f, hole / 2, field_radius * (1 - s / f)) for _, hole, s in stops)

    rows = []
    for ep in EYEPIECES:
        mag = f / ep.focal_mm
        pupil = d / mag
        tfov = math.degrees(ep.field_stop_mm / f)
        row = EyepieceRow(
            ep, mag, pupil, tfov, 2 * 100 * math.tan(math.radians(tfov / 2)), twilight_factor(d, mag),
            relative_brightness(pupil, 6.0), relative_brightness(pupil, EYE_PUPIL_MM),
            edge_light(ep.field_stop_mm / 2),
        )
        if pupil >= 5:
            row.notes.append("出瞳大，傍晚和月光下好用")
        if mag > PRACTICAL_MAX_X:
            row.notes.append(f"超过 {PRACTICAL_MAX_X} 倍，看地面景物会被热气流抖糊")
        if row.edge_light < 0.7:
            row.notes.append("视场边缘被正像棱镜挡暗，换 2 寸正像棱镜可以解决")
        rows.append(row)

    # 遮光环：只让成像需要的光锥通过，挡掉镜筒内壁反射的杂光
    liner_id = spec.tube_od_mm - 2 * spec.tube_wall_mm - 2 * (LINER_FIT_MM + LINER_WALL_MM)
    front_cone = light_cone_mm(d, f, spec.cell_rear_mm)
    if liner_id < front_cone + 1:
        warnings.append(f"内衬筒会挡光：镜筒前端光锥直径 {front_cone:.0f} mm，衬筒内径只有 {liner_id:.0f} mm，换粗一点的镜筒")
    baffles = []
    for k in range(1, spec.baffles + 1):
        pos = tube * k / (spec.baffles + 1)
        hole = light_cone_mm(d, f, spec.cell_rear_mm + pos) + 2  # 留 1 mm 余量
        if hole < liner_id - 4:
            baffles.append(Baffle(pos, hole))

    ca = chromatic_index(d, f)
    if spec.ed_glass:
        color = "ED 低色散玻璃：基本看不到紫边"
    elif ca < 3:
        color = "普通消色差：20 倍以上，亮的边缘会有紫边；换 ED 玻璃可以解决"
    elif ca < 5:
        color = "普通消色差：高倍时略有紫边"
    else:
        color = "普通消色差：几乎看不出紫边"

    # 重量和平衡（估计）：物镜和镜座在前，调焦座、棱镜、目镜在后
    tube_kg = math.pi * (spec.tube_od_mm - spec.tube_wall_mm) * spec.tube_wall_mm * tube * ALUMINIUM_KG_MM3
    masses = [
        (0.35, -spec.cell_rear_mm),  # 物镜和镜座
        (0.10, -spec.cell_rear_mm - 0.75 * d),  # 遮光罩
        (tube_kg, tube / 2),
        (0.45, tube + spec.focuser_length_mm / 2),  # 2 寸调焦座
        (0.15, tube + spec.focuser_length_mm + 40),  # 正像棱镜
        (0.30, tube + spec.focuser_length_mm + 80),  # 变焦目镜
    ]
    total = sum(m for m, _ in masses)
    balance = sum(m * z for m, z in masses) / total

    return SpottingDesign(spec, rows, min_focus, tube, 1.5 * d, baffles, ca, color, balance, total, warnings)


# ---------------------------------------------------------------- 物镜外协


@dataclass(frozen=True)
class ObjectiveSpec:
    aperture_mm: float
    focal_length_mm: float
    ed_glass: bool

    @property
    def lines(self) -> list[tuple[str, str]]:
        d, f = self.aperture_mm, self.focal_length_mm
        glass = ("ED 双分离：一片 ED 低色散玻璃（如成都光明 H-FK61，相当于 FPL-51）配一片镧系或重冕玻璃"
                 if self.ed_glass else "消色差双分离（如 H-K9L 冕牌 + H-ZF2 火石）")
        return [
            ("类型", glass + "，空气间隔或胶合均可"),
            ("通光口径", f"{d:g} mm（镜片外径按镜座设计，通光口径不小于 {d - 2:g} mm）"),
            ("焦距", f"{f:g} mm ±1%（F/{f / d:g}），请告知实测值"),
            ("像质", "@546 nm 波面 PV ≤ λ/4，斯特列尔比 ≥ 0.8；486–656 nm 色差校正"),
            ("镀膜", "每个面宽带多层增透膜，450–650 nm 平均反射率 ≤ 0.5%"),
            ("边缘", "镜片侧面涂黑，防止杂光"),
            ("交货", "装进镜座、调好间隔和偏心后交货（偏心 ≤ 0.02 mm）；镜座后端加工成和镜筒配合的止口"),
            ("用途", "看地面景物，最高约 60 倍；不需要平场镜"),
        ]


# ---------------------------------------------------------------- 3D 打印零件


@dataclass
class PrintedPart:
    name: str
    filename: str
    profile: stl.Profile
    note: str


def printed_parts(design: SpottingDesign, cell_od_mm: float | None = None) -> list[PrintedPart]:
    """遮光罩、镜头盖、带遮光环的内衬筒。全部用黑色 PETG 打印。"""
    spec = design.spec
    tube_id = spec.tube_od_mm - 2 * spec.tube_wall_mm
    cell_od = cell_od_mm or spec.tube_od_mm
    wall = 2.0
    shield_id = cell_od + 0.4
    parts = [
        PrintedPart("遮光罩", "dew_shield.stl", stl.tube_profile(shield_id / 2, shield_id / 2 + wall, design.dew_shield_mm),
                    f"套在物镜镜座外面（内径 {shield_id:.1f} mm），挡侧面来的光，防止镜片结露"),
        PrintedPart("镜头盖", "lens_cap.stl", stl.cap_profile(shield_id / 2 + wall + 0.2, wall, 15, 2),
                    "盖在遮光罩前端，不用时防尘"),
    ]
    outer_r = tube_id / 2 - LINER_FIT_MM
    inner_r = outer_r - LINER_WALL_MM
    z = 0.0
    for k, b in enumerate(design.baffles, 1):
        parts.append(PrintedPart(
            f"内衬筒 {k}（带遮光环）", f"liner_{k}.stl",
            stl.sleeve_with_baffle(inner_r, outer_r, b.position_mm - z, b.hole_mm / 2, 2.0),
            f"长 {b.position_mm - z:.0f} mm，后端遮光环孔径 {b.hole_mm:.0f} mm",
        ))
        z = b.position_mm
    parts.append(PrintedPart(
        f"内衬筒 {len(design.baffles) + 1}", f"liner_{len(design.baffles) + 1}.stl",
        stl.tube_profile(inner_r, outer_r, design.tube_length_mm - z),
        f"长 {design.tube_length_mm - z:.0f} mm，最后塞进去，把前面几节顶住",
    ))
    return parts
