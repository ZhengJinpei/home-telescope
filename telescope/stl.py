"""把旋转体零件（筒、环、盖子）做成 3D 打印用的 STL 文件。

零件用半截面轮廓描述：(r, z) 平面里的一个封闭多边形，r ≥ 0，绕 z 轴转一圈就是零件。
r = 0 的点在轴上（比如盖子的封口），只生成一个顶点，网格依然封闭。
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

Profile = list[tuple[float, float]]


@dataclass
class Mesh:
    vertices: list[tuple[float, float, float]]
    triangles: list[tuple[int, int, int]]

    def volume(self) -> float:
        """有符号体积（法线朝外时为正）。"""
        v = self.vertices
        total = 0.0
        for a, b, c in self.triangles:
            (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = v[a], v[b], v[c]
            total += x1 * (y2 * z3 - y3 * z2) - x2 * (y1 * z3 - y3 * z1) + x3 * (y1 * z2 - y2 * z1)
        return total / 6

    def is_watertight(self) -> bool:
        """每条边正好被两个三角形共用，而且方向相反（3D 打印切片软件要求的“封闭”）。"""
        edges: dict[tuple[int, int], int] = {}
        for a, b, c in self.triangles:
            for e in ((a, b), (b, c), (c, a)):
                edges[e] = edges.get(e, 0) + 1
        return all(n == 1 and edges.get((v, u)) == 1 for (u, v), n in edges.items())


def revolve(profile: Profile, segments: int = 128) -> Mesh:
    if len(profile) < 3:
        raise ValueError("轮廓至少要 3 个点")
    if any(r < 0 for r, _ in profile):
        raise ValueError("轮廓的 r 不能是负数")
    vertices: list[tuple[float, float, float]] = []
    ring: list[list[int]] = []  # ring[i][j]：轮廓第 i 个点转到第 j 个角度时的顶点编号
    for r, z in profile:
        if r == 0:
            vertices.append((0.0, 0.0, z))
            ring.append([len(vertices) - 1] * segments)
        else:
            start = len(vertices)
            for j in range(segments):
                t = 2 * math.pi * j / segments
                vertices.append((r * math.cos(t), r * math.sin(t), z))
            ring.append(list(range(start, start + segments)))
    triangles = []
    n = len(profile)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        for j in range(segments):
            k = (j + 1) % segments
            for tri in ((a[j], b[j], b[k]), (a[j], b[k], a[k])):
                if len(set(tri)) == 3:  # 轴上的点会让一半三角形退化，跳过
                    triangles.append(tri)
    mesh = Mesh(vertices, triangles)
    if mesh.volume() < 0:
        mesh.triangles = [(a, c, b) for a, b, c in mesh.triangles]
    return mesh


def profile_volume(profile: Profile) -> float:
    """帕普斯定理：旋转体体积 = 2π × 截面积 × 截面形心到轴的距离（理想圆，不是多边形）。"""
    area = moment = 0.0
    for (r1, z1), (r2, z2) in zip(profile, profile[1:] + profile[:1]):
        cross = r1 * z2 - r2 * z1
        area += cross
        moment += (r1 + r2) * cross
    return abs(2 * math.pi * moment / 6)


def write_stl(mesh: Mesh, path: Path, name: str = "part") -> None:
    """二进制 STL，单位 mm。"""
    v = mesh.vertices
    with open(path, "wb") as f:
        f.write(name.encode("ascii", "replace")[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(mesh.triangles)))
        for a, b, c in mesh.triangles:
            p, q, s = v[a], v[b], v[c]
            u = (q[0] - p[0], q[1] - p[1], q[2] - p[2])
            w = (s[0] - p[0], s[1] - p[1], s[2] - p[2])
            nx, ny, nz = u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]
            length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            f.write(struct.pack("<12fH", nx / length, ny / length, nz / length, *p, *q, *s, 0))


# ---------------------------------------------------------------- 常用轮廓


def tube_profile(inner_r: float, outer_r: float, length: float) -> Profile:
    """一段圆筒。"""
    return [(inner_r, 0.0), (outer_r, 0.0), (outer_r, length), (inner_r, length)]


def sleeve_with_baffle(inner_r: float, outer_r: float, length: float, hole_r: float, thickness: float) -> Profile:
    """衬筒，后端带一圈遮光环（中间开 hole_r 的孔）。"""
    return [(inner_r, 0.0), (outer_r, 0.0), (outer_r, length), (hole_r, length),
            (hole_r, length - thickness), (inner_r, length - thickness)]


def cap_profile(inner_r: float, wall: float, depth: float, end: float) -> Profile:
    """杯形盖子：封口在 z = 0 一端，开口朝上。"""
    return [(0.0, 0.0), (inner_r + wall, 0.0), (inner_r + wall, depth + end),
            (inner_r, depth + end), (inner_r, end), (0.0, end)]
