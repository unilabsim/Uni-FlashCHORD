"""Device-native friction-cone wrench support evaluation."""

from __future__ import annotations

from typing import Any, cast

import torch
import warp as wp


@wp.func
def normalized_vector(value: Any) -> Any:
    return value / wp.length(value)


@wp.kernel
def wrench_support_kernel(
    position_o: Any,
    direction_o: Any,
    basis: Any,
    radius: Any,
    support: Any,
    num_hands: int,
    num_bodies: int,
    num_contacts: int,
    num_basis: int,
    friction: float,
    cone_edges: int,
) -> None:
    tid = cast(Any, wp.tid())
    basis_index = tid % num_basis
    body = (tid // num_basis) % num_bodies
    hand = (tid // (num_basis * num_bodies)) % num_hands
    world = tid // (num_basis * num_bodies * num_hands)
    contact_start = ((world * num_hands + hand) * num_bodies + body) * num_contacts
    direction = basis[body * num_basis + basis_index]
    direction_f = wp.spatial_top(direction)
    direction_t = wp.spatial_bottom(direction)
    best = wp.float32(0.0)
    for contact_index in range(num_contacts):
        slot = contact_start + contact_index
        normal_length = wp.length(direction_o[slot])
        if normal_length <= 1.0e-3:
            continue
        normal = direction_o[slot] / normal_length
        position = position_o[slot]
        sign = wp.where(normal[2] >= 0.0, 1.0, -1.0)
        a = -1.0 / (sign + normal[2])
        bb = normal[0] * normal[1] * a
        tangent1 = normalized_vector(
            wp.vec3(
                1.0 + sign * normal[0] * normal[0] * a,
                sign * bb,
                -sign * normal[0],
            )
        )
        tangent2 = normalized_vector(
            wp.vec3(bb, sign + normal[1] * normal[1] * a, -normal[1])
        )
        for edge in range(cone_edges + 1):
            if edge < cone_edges:
                theta = float(edge) * (2.0 * wp.pi / float(cone_edges))
                force = normalized_vector(
                    normal
                    + friction * (wp.cos(theta) * tangent1 + wp.sin(theta) * tangent2)
                )
            else:
                force = normal
            torque = wp.cross(position, force) / radius[body]
            value = wp.dot(direction_f, force) + wp.dot(direction_t, torque)
            best = wp.max(best, value)
    support[tid] = best


_WARP_INITIALIZED = False


def wrench_support_cuda(
    position_o: torch.Tensor,
    direction_o: torch.Tensor,
    basis: torch.Tensor,
    radius: torch.Tensor,
    *,
    friction: float,
    cone_edges: int,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate support directly into a persistent Torch output on CUDA."""
    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True
    n, h, b, contacts, _ = position_o.shape
    directions = basis.shape[1]
    if output is None:
        output = position_o.new_zeros((n, h, b, directions))
    elif tuple(output.shape) != (n, h, b, directions):
        raise ValueError(
            f"wrench support output shape {tuple(output.shape)} does not match "
            f"{(n, h, b, directions)}"
        )
    tensors = (position_o, direction_o, basis, radius, output)
    if any(tuple(tensor.shape) and not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("CUDA wrench support requires contiguous tensors")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise TypeError("CUDA wrench support requires float32 tensors")
    device = wp.device_from_torch(position_o.device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(position_o.device))
    wp.launch(
        wrench_support_kernel,
        dim=n * h * b * directions,
        inputs=[
            wp.from_torch(position_o.reshape(-1, 3), dtype=cast(Any, wp.vec3)),
            wp.from_torch(direction_o.reshape(-1, 3), dtype=cast(Any, wp.vec3)),
            wp.from_torch(basis.reshape(-1, 6), dtype=cast(Any, wp.spatial_vector)),
            wp.from_torch(radius),
            wp.from_torch(output.reshape(-1)),
            h,
            b,
            contacts,
            directions,
            float(friction),
            int(cone_edges),
        ],
        device=device,
        stream=stream,
    )
    return output
