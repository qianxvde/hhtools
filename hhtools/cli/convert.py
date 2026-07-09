"""``hhtools convert`` — batch convert mocap files into the unified NPZ."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from hhtools.core.resample import resample_motion
from hhtools.io import bvh, npz
from hhtools.io.base import load_motion

app = typer.Typer(no_args_is_help=True, help="Convert BVH / GLB to the unified NPZ.")
_console = Console()


@app.command("run")
def run_convert(
    inputs: list[Path] = typer.Argument(..., help="Files or directories to convert."),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory for NPZ files."),
    unit: str = typer.Option("cm", "--unit", help="Source unit for BVH files (cm, mm, m, ...)."),
    target_up_axis: str = typer.Option(
        "Z", "--up", case_sensitive=False, help="Internal up-axis (Z or Y)."
    ),
    target_fps: float | None = typer.Option(
        None, "--fps", help="Optional target framerate (resamples with SLERP)."
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing NPZ files."),
) -> None:
    """Convert one or more motion files into the unified NPZ schema."""
    files = _collect_input_files(inputs, recursive=recursive)
    if not files:
        _console.print("[yellow]No supported motion files found.[/]")
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)
    _console.print(f"Converting [bold]{len(files)}[/] file(s) into [bold]{out}[/]")

    ok: list[Path] = []
    errors: list[tuple[Path, str]] = []

    with Progress(console=_console) as progress:
        task: TaskID = progress.add_task("convert", total=len(files))
        for src in files:
            try:
                motion = _load_one(src, unit=unit, target_up_axis=target_up_axis)
                if target_fps is not None and abs(target_fps - motion.framerate) > 1e-4:
                    motion = resample_motion(motion, target_fps)
                dst = out / (src.stem + ".npz")
                if dst.exists() and not overwrite:
                    errors.append((src, f"destination exists ({dst}); use --overwrite"))
                else:
                    npz.save_npz(motion, dst)
                    ok.append(dst)
            except Exception as exc:  # pragma: no cover - surfaced in CLI output
                errors.append((src, str(exc)))
            finally:
                progress.advance(task)

    table = Table(title="convert summary", show_lines=False)
    table.add_column("status", style="bold")
    table.add_column("path")
    for p in ok:
        table.add_row("[green]ok[/]", str(p))
    for src, msg in errors:
        table.add_row("[red]fail[/]", f"{src}  —  {msg}")
    _console.print(table)

    if errors and not ok:
        raise typer.Exit(code=2)


def _collect_input_files(inputs: list[Path], *, recursive: bool) -> list[Path]:
    extensions = {".bvh", ".glb", ".gltf", ".npz"}
    out: list[Path] = []
    for item in inputs:
        if item.is_file():
            if item.suffix.lower() in extensions:
                out.append(item)
        elif item.is_dir():
            iterator = item.rglob("*") if recursive else item.glob("*")
            for child in iterator:
                if child.is_file() and child.suffix.lower() in extensions:
                    out.append(child)
        else:
            _console.print(f"[yellow]skip: {item} does not exist[/]")
    return sorted(set(out))


def _load_one(src: Path, *, unit: str, target_up_axis: str):  # type: ignore[no-untyped-def]
    ext = src.suffix.lower()
    if ext == ".bvh":
        return bvh.load_bvh(src, unit=unit, target_up_axis=target_up_axis)
    return load_motion(src)


# --------------------------------------------------------------------------
# Motion-data conversion (mjlab NPZ) — single source of truth, headless parity
# with the 数据转换 web panel. Delegates to ``hhtools.dataconvert``.
# --------------------------------------------------------------------------


@app.command("csv-to-npz")
def csv_to_npz(
    source: Path = typer.Argument(..., help="Retarget CSV/PKL file (or a directory of them)."),
    mjcf: Path = typer.Option(..., "--mjcf", help="Target robot MJCF/xml."),
    out: Path = typer.Option(..., "--out", "-o", help="Output NPZ file or directory."),
    body_states: bool = typer.Option(True, "--body-states/--no-body-states", help="Compute MuJoCo-FK body world poses."),
    snap_ground: bool = typer.Option(False, "--snap-ground", help="Snap the lowest foot box to the floor."),
    fps: float | None = typer.Option(None, "--fps", help="Override the source framerate."),
) -> None:
    """Convert retarget CSV/PKL trajectories into canonical mjlab NPZ."""
    from hhtools.dataconvert.convert import ConvertOptions, convert_file, npz_payload_summary

    sources = _collect_traj_files(source)
    if not sources:
        _console.print("[yellow]No .csv/.pkl trajectory files found.[/]")
        raise typer.Exit(code=1)
    out_is_dir = out.suffix.lower() != ".npz" or len(sources) > 1
    options = ConvertOptions(compute_body_states=body_states, snap_to_ground=snap_ground)
    ok = 0
    for src in sources:
        dst = (out / f"{src.stem}.npz") if out_is_dir else out
        try:
            payload = convert_file(src, mjcf, dst, options=options, fps_override=fps)
            s = npz_payload_summary(payload)
            _console.print(f"[green]ok[/] {src.name} -> {dst} ({s['frames']} frames, {s['num_joints']} joints)")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            _console.print(f"[red]fail[/] {src.name} — {exc}")
    if ok == 0:
        raise typer.Exit(code=2)


@app.command("check")
def check(
    npz_path: Path = typer.Argument(..., help="Converted mjlab NPZ."),
    mjcf: Path = typer.Option(..., "--mjcf", help="Robot MJCF/xml to replay against."),
    threshold: float = typer.Option(0.001, "--threshold", help="Contact distance threshold (m)."),
) -> None:
    """Audit a motion NPZ for self-collision / ground penetration / contacts."""
    import numpy as np

    from hhtools.dataconvert.contacts import audit
    from hhtools.dataconvert.mjcf_model import MjcfRobot

    with np.load(npz_path, allow_pickle=True) as archive:
        payload = {k: archive[k] for k in archive.files}
    robot = MjcfRobot.from_path(mjcf)
    report, has_issues = audit(robot, payload, threshold=threshold)
    _console.print(report)
    if has_issues:
        raise typer.Exit(code=2)


@app.command("speeds")
def speeds(
    npz_path: Path = typer.Argument(..., help="Converted mjlab NPZ."),
) -> None:
    """Print a root-velocity / yaw-rate summary for a motion NPZ."""
    from hhtools.dataconvert.speeds import summarize_file

    summary = summarize_file(npz_path)
    table = Table(title=f"speeds — {npz_path.name}")
    table.add_column("metric", style="bold")
    table.add_column("value")
    for key, val in summary.as_dict().items():
        table.add_row(key, f"{val:.4f}" if isinstance(val, float) else str(val))
    _console.print(table)


@app.command("fullstate")
def fullstate(
    source: Path = typer.Argument(..., help="Booster full_state JSON clip (or directory)."),
    mjcf: Path = typer.Option(..., "--mjcf", help="Target robot MJCF/xml."),
    out: Path = typer.Option(..., "--out", "-o", help="Output NPZ file or directory."),
    joints: str = typer.Option(..., "--joints", help="Comma-separated source DOF order (full_state dof block)."),
    body_states: bool = typer.Option(True, "--body-states/--no-body-states"),
) -> None:
    """Import Booster full_state TXT/JSON clips, then convert to mjlab NPZ."""
    from hhtools.dataconvert.convert import ConvertOptions, convert_trajectory, npz_payload_summary, save_npz
    from hhtools.dataconvert.fullstate import load_fullstate
    from hhtools.dataconvert.mjcf_model import MjcfRobot

    source_joints = tuple(j.strip() for j in joints.split(",") if j.strip())
    sources = sorted(source.glob("*.txt")) + sorted(source.glob("*.json")) if source.is_dir() else [source]
    if not sources:
        _console.print("[yellow]No full_state clips found.[/]")
        raise typer.Exit(code=1)
    robot = MjcfRobot.from_path(mjcf)
    out_is_dir = out.suffix.lower() != ".npz" or len(sources) > 1
    options = ConvertOptions(compute_body_states=body_states)
    ok = 0
    for src in sources:
        dst = (out / f"{src.stem}.npz") if out_is_dir else out
        try:
            traj = load_fullstate(src, source_joints)
            payload = convert_trajectory(traj, robot, options)
            save_npz(dst, payload)
            s = npz_payload_summary(payload)
            _console.print(f"[green]ok[/] {src.name} -> {dst} ({s['frames']} frames)")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            _console.print(f"[red]fail[/] {src.name} — {exc}")
    if ok == 0:
        raise typer.Exit(code=2)


@app.command("csv-to-txt")
def csv_to_txt(
    source: Path = typer.Argument(..., help="Retarget CSV/PKL file (or a directory of them)."),
    mjcf: Path = typer.Option(..., "--mjcf", help="Target robot MJCF/xml (used for end-effector FK)."),
    out: Path = typer.Option(..., "--out", "-o", help="Output .txt file or directory."),
    joints: str = typer.Option(
        "", "--joints", help="Comma-separated joint order (env motion_joint_names). Overrides --profile."
    ),
    end_effectors: str = typer.Option(
        "", "--end-effectors", help="Comma-separated end-effector body names. Defaults to the profile / Booster default."
    ),
    profile: str = typer.Option(
        "", "--profile", help="Export profile id supplying joint order + end effectors (e.g. booster_isaaclab.t1.amp_txt)."
    ),
    fps: float | None = typer.Option(None, "--fps", help="Override the source framerate."),
) -> None:
    """Convert retarget CSV/PKL trajectories into booster_isaaclab AMP .txt clips."""
    from hhtools.dataconvert.isaaclab_txt import (
        DEFAULT_END_EFFECTOR_BODIES,
        IsaacLabTxtOptions,
        convert_file_to_amp_txt,
    )
    from hhtools.dataconvert.mjcf_model import MjcfRobot
    from hhtools.dataconvert import profiles as _profiles

    joint_order: tuple[str, ...] = tuple(j.strip() for j in joints.split(",") if j.strip())
    ee_bodies: tuple[str, ...] = tuple(b.strip() for b in end_effectors.split(",") if b.strip())
    if profile:
        prof = _profiles.get_profile(profile)
        joint_order = joint_order or prof.joint_order
        ee_bodies = ee_bodies or prof.end_effector_bodies
    if not joint_order:
        # Derive the AMP joint order from the robot's own actuated-joint order.
        joint_order = tuple(MjcfRobot.from_path(mjcf).joint_names)
    if not joint_order:
        _console.print("[red]Could not determine a joint order (empty robot?). Pass --joints.[/]")
        raise typer.Exit(code=1)
    options = IsaacLabTxtOptions(
        joint_order=joint_order,
        end_effector_bodies=ee_bodies or DEFAULT_END_EFFECTOR_BODIES,
    )

    sources = _collect_traj_files(source)
    if not sources:
        _console.print("[yellow]No .csv/.pkl trajectory files found.[/]")
        raise typer.Exit(code=1)
    out_is_dir = out.suffix.lower() != ".txt" or len(sources) > 1
    ok = 0
    for src in sources:
        dst = (out / f"{src.stem}.txt") if out_is_dir else out
        try:
            s = convert_file_to_amp_txt(src, mjcf, dst, options=options, fps_override=fps)
            _console.print(
                f"[green]ok[/] {src.name} -> {dst} ({s['frames']} frames, obs_dim={s['observation_dim']})"
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001
            _console.print(f"[red]fail[/] {src.name} — {exc}")
    if ok == 0:
        raise typer.Exit(code=2)


@app.command("export")
def export(
    source: Path = typer.Argument(..., help="Retarget CSV/PKL file (or a directory of them)."),
    mjcf: Path = typer.Option(..., "--mjcf", help="Target robot MJCF/xml (FK + contact model)."),
    out: Path = typer.Option(..., "--out", "-o", help="Output file or directory."),
    profile: str = typer.Option(..., "--profile", help="Training export profile id."),
    snap_ground: bool = typer.Option(False, "--snap-ground", help="NPZ only: snap the lowest foot box to the floor."),
    fps: float | None = typer.Option(None, "--fps", help="Override the source framerate."),
) -> None:
    """Export a trajectory to a training target via a profile (npz or amp_txt)."""
    from hhtools.dataconvert.csv_io import load_trajectory
    from hhtools.dataconvert.mjcf_model import MjcfRobot
    from hhtools.dataconvert import profiles as _profiles

    prof = _profiles.get_profile(profile)
    ext = prof.default_ext()
    robot = MjcfRobot.from_path(mjcf)
    sources = _collect_traj_files(source)
    if not sources:
        _console.print("[yellow]No .csv/.pkl trajectory files found.[/]")
        raise typer.Exit(code=1)
    out_is_dir = out.suffix.lower() != ext or len(sources) > 1
    ok = 0
    for src_path in sources:
        dst = (out / f"{src_path.stem}{ext}") if out_is_dir else out
        try:
            traj = load_trajectory(src_path, fps_override=fps)
            s = _profiles.export_with_profile(traj, robot, prof, dst, snap_to_ground=snap_ground)
            _console.print(f"[green]ok[/] {src_path.name} -> {dst} ({s.get('frames')} frames)")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            _console.print(f"[red]fail[/] {src_path.name} — {exc}")
    if ok == 0:
        raise typer.Exit(code=2)


@app.command("profiles")
def profiles_cmd() -> None:
    """List the registered training-export profiles."""
    from hhtools.dataconvert import profiles as _profiles

    table = Table(title="training export profiles")
    table.add_column("id", style="bold")
    table.add_column("framework")
    table.add_column("format")
    table.add_column("output_subdir")
    for prof in _profiles.list_profiles():
        table.add_row(prof.id, prof.framework, prof.fmt, prof.output_subdir)
    _console.print(table)


def _collect_traj_files(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted([*source.glob("*.csv"), *source.glob("*.pkl"), *source.glob("*.pickle")])
    return [source] if source.is_file() else []


__all__ = ["app"]
