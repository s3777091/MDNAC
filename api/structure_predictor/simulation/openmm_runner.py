from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class OpenMMSimulationEnvironment:
    solvent: str = "water"
    temperature_k: float = 300.0
    ph: float = 7.4
    salt_m: float = 0.15
    steps: int = 50_000
    report_interval: int = 500
    gpu_device: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OpenMMSimulationEnvironment":
        environment = cls(
            solvent=str(payload.get("solvent") or "water"),
            temperature_k=float(payload.get("temperature_k", 300.0)),
            ph=float(payload.get("ph", 7.4)),
            salt_m=float(payload.get("salt_m", 0.15)),
            steps=int(payload.get("steps", 50_000)),
            report_interval=int(payload.get("report_interval", 500)),
            gpu_device=int(payload.get("gpu_device", 0)),
        )
        environment.validate()
        return environment

    def validate(self) -> None:
        if self.solvent != "water":
            raise ValueError("Only solvent='water' is supported for OpenMM simulation.")
        if self.temperature_k <= 0:
            raise ValueError("temperature_k must be greater than 0.")
        if not 0 <= self.ph <= 14:
            raise ValueError("ph must be between 0 and 14.")
        if self.salt_m < 0:
            raise ValueError("salt_m must be greater than or equal to 0.")
        if self.steps <= 0:
            raise ValueError("steps must be greater than 0.")
        if self.report_interval <= 0:
            raise ValueError("report_interval must be greater than 0.")
        if self.report_interval > self.steps:
            raise ValueError("report_interval must be less than or equal to steps.")
        if self.gpu_device < 0:
            raise ValueError("gpu_device must be greater than or equal to 0.")


class OpenMMSimulationRunner:
    def __init__(self, jobs_root: str | Path) -> None:
        self.jobs_root = Path(jobs_root).expanduser().resolve()

    def run(
        self,
        payload: dict[str, Any],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        environment = OpenMMSimulationEnvironment.from_payload(payload.get("environment") or {})
        pdb_path = payload.get("pdb_path")
        cif_path = payload.get("cif_path")
        if not pdb_path:
            if cif_path:
                raise ValueError("CIF simulation is not supported yet; please provide pdb_path.")
            raise ValueError("pdb_path is required for OpenMM simulation.")

        input_pdb_path = Path(str(pdb_path)).expanduser()
        if not input_pdb_path.is_file():
            raise FileNotFoundError(f"PDB input file not found: {input_pdb_path}")

        job_id = str(payload["job_id"])
        output_dir = self.jobs_root / job_id / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        initial_pdb = output_dir / "simulation_initial.pdb"
        final_pdb = output_dir / "simulation_final.pdb"
        trajectory_dcd = output_dir / "trajectory.dcd"
        state_csv = output_dir / "state.csv"

        openmm = _import_openmm()
        _report_progress(progress_callback, 0, environment.steps, "preparation")

        pdb = openmm["PDBFile"](str(input_pdb_path))
        forcefield = openmm["ForceField"]("amber14-all.xml", "amber14/tip3pfb.xml")
        modeller = openmm["Modeller"](pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield, pH=environment.ph)
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=1.0 * openmm["nanometer"],
            ionicStrength=environment.salt_m * openmm["molar"],
        )
        with initial_pdb.open("w", encoding="utf-8") as handle:
            openmm["PDBFile"].writeFile(modeller.topology, modeller.positions, handle)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=openmm["PME"],
            nonbondedCutoff=1.0 * openmm["nanometer"],
            constraints=openmm["HBonds"],
        )
        integrator = openmm["LangevinMiddleIntegrator"](
            environment.temperature_k * openmm["kelvin"],
            1.0 / openmm["picosecond"],
            0.002 * openmm["picoseconds"],
        )
        platform = openmm["Platform"].getPlatformByName("CUDA")
        platform_properties = {
            "DeviceIndex": str(environment.gpu_device),
            "Precision": "mixed",
        }
        simulation = openmm["Simulation"](
            modeller.topology,
            system,
            integrator,
            platform,
            platform_properties,
        )
        simulation.context.setPositions(modeller.positions)
        _report_progress(progress_callback, 0, environment.steps, "minimization")
        simulation.minimizeEnergy()

        simulation.reporters.append(
            openmm["DCDReporter"](str(trajectory_dcd), environment.report_interval)
        )
        simulation.reporters.append(
            openmm["StateDataReporter"](
                str(state_csv),
                environment.report_interval,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                speed=True,
                separator=",",
            )
        )

        completed_steps = 0
        while completed_steps < environment.steps:
            chunk_steps = min(environment.report_interval, environment.steps - completed_steps)
            simulation.step(chunk_steps)
            completed_steps += chunk_steps
            _report_progress(
                progress_callback,
                completed_steps,
                environment.steps,
                "simulation",
            )

        state = simulation.context.getState(getPositions=True)
        with final_pdb.open("w", encoding="utf-8") as handle:
            openmm["PDBFile"].writeFile(simulation.topology, state.getPositions(), handle)

        return {
            "initial_pdb": str(initial_pdb.resolve()),
            "final_pdb": str(final_pdb.resolve()),
            "trajectory_dcd": str(trajectory_dcd.resolve()),
            "state_csv": str(state_csv.resolve()),
            "steps": environment.steps,
            "temperature_k": environment.temperature_k,
            "ph": environment.ph,
            "salt_m": environment.salt_m,
            "gpu_device": environment.gpu_device,
        }


def _report_progress(
    progress_callback: ProgressCallback | None,
    current_step: int,
    total_steps: int,
    step: str,
) -> None:
    if progress_callback is not None:
        progress_callback(current_step, total_steps, step)


def _import_openmm() -> dict[str, Any]:
    try:
        from openmm import LangevinMiddleIntegrator, Platform
        from openmm.app import DCDReporter, ForceField, HBonds, PME, Modeller, PDBFile, Simulation
        from openmm.app import StateDataReporter
        from openmm.unit import kelvin, molar, nanometer, picosecond, picoseconds
    except ImportError:
        try:
            from simtk.openmm import LangevinMiddleIntegrator, Platform
            from simtk.openmm.app import DCDReporter, ForceField, HBonds, PME, Modeller
            from simtk.openmm.app import PDBFile, Simulation, StateDataReporter
            from simtk.unit import kelvin, molar, nanometer, picosecond, picoseconds
        except ImportError as exc:
            raise RuntimeError(
                "Running OpenMM simulations requires `openmm` with CUDA support installed "
                "in the Celery worker environment."
            ) from exc

    return {
        "DCDReporter": DCDReporter,
        "ForceField": ForceField,
        "HBonds": HBonds,
        "LangevinMiddleIntegrator": LangevinMiddleIntegrator,
        "Modeller": Modeller,
        "PDBFile": PDBFile,
        "PME": PME,
        "Platform": Platform,
        "Simulation": Simulation,
        "StateDataReporter": StateDataReporter,
        "kelvin": kelvin,
        "molar": molar,
        "nanometer": nanometer,
        "picosecond": picosecond,
        "picoseconds": picoseconds,
    }
