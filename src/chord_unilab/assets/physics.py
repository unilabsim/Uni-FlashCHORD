"""Backend-neutral physics and collision configuration for assembled assets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactPhysics:
    solref: tuple[float, float] = (0.02, 1.58113898)
    solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    margin: float = 0.0
    gap: float = 0.0


@dataclass(frozen=True)
class SolverPhysics:
    timestep: float = 0.01
    solver: str = "Newton"
    integrator: str = "implicitfast"
    cone: str = "pyramidal"
    impratio: float = 20.0
    iterations: int = 100
    ls_iterations: int = 50

    def as_mjcf(self) -> dict[str, str]:
        return {
            "timestep": f"{self.timestep:.9g}",
            "solver": self.solver,
            "integrator": self.integrator,
            "cone": self.cone,
            "impratio": f"{self.impratio:.9g}",
            "iterations": str(self.iterations),
            "ls_iterations": str(self.ls_iterations),
        }


@dataclass(frozen=True)
class CollisionPhysics:
    object_group: int = 1 << 13
    support_group: int = 1 << 1
    ground_group: int = 1 << 14
    all_groups: int = (1 << 31) - 1

    def conaffinity(self, group: int) -> int:
        if group <= 0 or group & (group - 1):
            raise ValueError(f"collision group must be one positive bit, got {group}")
        return self.all_groups & ~group


@dataclass(frozen=True)
class ScenePhysics:
    contact: ContactPhysics = ContactPhysics()
    ground_contact: ContactPhysics = ContactPhysics(solref=(0.02, 1.0))
    solver: SolverPhysics = SolverPhysics()
    collision: CollisionPhysics = CollisionPhysics()
    joint_solreflimit: tuple[float, float] = (-10000.0, -10.0)
    joint_solimplimit: tuple[float, float, float, float, float] = (
        0.9,
        0.95,
        0.001,
        0.5,
        2.0,
    )


DEFAULT_SCENE_PHYSICS = ScenePhysics()


def contact_mjcf(contact: ContactPhysics) -> dict[str, str]:
    return {
        "solref": " ".join(f"{value:.9g}" for value in contact.solref),
        "solimp": " ".join(f"{value:.9g}" for value in contact.solimp),
        "friction": " ".join(f"{value:.9g}" for value in contact.friction),
        "margin": f"{contact.margin:.9g}",
        "gap": f"{contact.gap:.9g}",
    }
