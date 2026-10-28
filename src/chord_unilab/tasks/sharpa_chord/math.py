"""Torch rotation primitives using CHORD's wxyz convention."""

from __future__ import annotations

import torch


def normalize(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )


def multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    scalar = quat[..., :1]
    return vector + 2.0 * torch.linalg.cross(
        xyz, torch.linalg.cross(xyz, vector) + scalar * vector
    )


def rotate_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return rotate(conjugate(quat), vector)


def unique(quat: torch.Tensor) -> torch.Tensor:
    return torch.where(quat[..., :1] < 0.0, -quat, quat)


def from_euler_xyz(euler: torch.Tensor) -> torch.Tensor:
    half_roll = 0.5 * euler[..., 0]
    half_pitch = 0.5 * euler[..., 1]
    half_yaw = 0.5 * euler[..., 2]
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    quat = unique(normalize(quat))
    vector = quat[..., 1:]
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sine, quat[..., :1])
    scale = torch.where(sine < 1.0e-8, 2.0, angle / sine.clamp_min(1.0e-12))
    return vector * scale


def geodesic_angle(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    dot = (normalize(left) * normalize(right)).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)
