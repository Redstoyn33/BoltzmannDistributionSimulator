import csv
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


@dataclass
class SimulationParameters:
    n_particles: int = 800
    sphere_radius: float = 12.0
    dt: float = 0.002
    steps_per_frame: int = 2
    particle_mass: float = 1.0
    temperature: float = 1.0
    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff_sigma: float = 2.5
    wall_restitution: float = 1.0
    seed: int | None = None
    init_min_distance_sigma: float = 0.9
    tracked_particle_index: int = 0
    save_every: int = 10
    max_trajectory_points: int = 500


class MolecularDynamicsLJ:
    def __init__(self, params: SimulationParameters) -> None:
        self.params = params
        self.rng = np.random.default_rng(params.seed)

        self.k_b = 1.0  # reduced units
        self.n_particles = params.n_particles
        self.radius = float(params.sphere_radius)

        self.mass = float(params.particle_mass)
        self.masses = np.full(self.n_particles, self.mass, dtype=float)

        self.positions = np.zeros((self.n_particles, 3), dtype=float)
        self.velocities = np.zeros((self.n_particles, 3), dtype=float)
        self.forces = np.zeros((self.n_particles, 3), dtype=float)

        self.time = 0.0
        self.step_index = 0

        self.epsilon = float(params.epsilon)
        self.sigma = float(params.sigma)
        self.cutoff = float(params.cutoff_sigma) * self.sigma
        self.cutoff_sq = self.cutoff * self.cutoff

        sigma_over_rc_6 = (self.sigma / self.cutoff) ** 6
        self.potential_shift = (
            4.0 * self.epsilon * (sigma_over_rc_6**2 - sigma_over_rc_6)
        )

        self.potential_energy = 0.0
        self.virial = 0.0

        self.energy_log: list[tuple[float, float, float, float]] = []
        self.temperature_log: list[tuple[float, float]] = []
        self.pressure_log: list[tuple[float, float]] = []
        self.track_log: list[tuple[float, float, float, float]] = []

        self._initialize_positions()
        self._initialize_velocities_maxwell_boltzmann()
        self.forces, self.potential_energy, self.virial = self.compute_forces()
        self._log_state()

    def _random_point_in_sphere(self, margin: float = 0.0) -> np.ndarray:
        available_radius = self.radius - margin
        while True:
            point = self.rng.uniform(-available_radius, available_radius, size=3)
            if np.dot(point, point) <= available_radius**2:
                return point

    def _initialize_positions(self) -> None:
        min_distance = self.params.init_min_distance_sigma * self.sigma
        min_distance_sq = min_distance * min_distance

        for i in range(self.n_particles):
            placed = False
            for _ in range(20000):
                candidate = self._random_point_in_sphere(margin=0.2 * self.sigma)

                if i == 0:
                    candidate = np.array([0.0, 0.0, 0.0], dtype=float)

                valid = True
                for j in range(i):
                    delta = candidate - self.positions[j]
                    if np.dot(delta, delta) < min_distance_sq:
                        valid = False
                        break

                if valid:
                    self.positions[i] = candidate
                    placed = True
                    break

            if not placed:
                raise RuntimeError(
                    "Could not place particles without strong overlap. "
                    "Decrease N, increase sphere radius, or reduce init_min_distance_sigma."
                )

    def _initialize_velocities_maxwell_boltzmann(self) -> None:
        sigma_v = np.sqrt(self.k_b * self.params.temperature / self.mass)
        self.velocities = self.rng.normal(0.0, sigma_v, size=(self.n_particles, 3))

        center_of_mass_velocity = np.mean(self.velocities, axis=0)
        self.velocities -= center_of_mass_velocity

        kinetic = self.kinetic_energy()
        dof = 3 * self.n_particles - 3
        if dof > 0 and kinetic > 0:
            current_temperature = 2.0 * kinetic / dof
            scale = np.sqrt(self.params.temperature / current_temperature)
            self.velocities *= scale

    def _lj_force_energy_virial(
        self,
        delta: np.ndarray,
        distance_sq: float,
    ) -> tuple[np.ndarray, float, float]:
        inv_r2 = 1.0 / distance_sq
        sigma_r_sq = (self.sigma * self.sigma) * inv_r2
        sigma_r_6 = sigma_r_sq**3
        sigma_r_12 = sigma_r_6**2

        potential = (
            4.0 * self.epsilon * (sigma_r_12 - sigma_r_6) - self.potential_shift
        )

        coefficient = (
            24.0 * self.epsilon * (2.0 * sigma_r_12 - sigma_r_6) * inv_r2
        )
        force_vector = coefficient * delta

        virial_pair = float(np.dot(delta, force_vector))
        return force_vector, potential, virial_pair

    def compute_forces(self) -> tuple[np.ndarray, float, float]:
        forces = np.zeros_like(self.positions)
        total_potential = 0.0
        total_virial = 0.0

        box_min = -self.radius
        box_max = self.radius

        cell_size = self.cutoff
        n_cells = max(1, int(np.floor((box_max - box_min) / cell_size)))
        cell_size = (box_max - box_min) / n_cells

        scaled = (self.positions - box_min) / cell_size
        cell_ids = np.floor(scaled).astype(int)
        cell_ids = np.clip(cell_ids, 0, n_cells - 1)

        cells: dict[tuple[int, int, int], list[int]] = {}
        for i in range(self.n_particles):
            key = tuple(cell_ids[i])
            cells.setdefault(key, []).append(i)

        neighbor_shifts = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]

        visited_cell_pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()

        for cell_key, particle_list in cells.items():
            cx, cy, cz = cell_key

            for dx, dy, dz in neighbor_shifts:
                neighbor_key = (cx + dx, cy + dy, cz + dz)
                if neighbor_key not in cells:
                    continue

                cell_pair = tuple(sorted((cell_key, neighbor_key)))
                if cell_pair in visited_cell_pairs:
                    continue
                visited_cell_pairs.add(cell_pair)

                neighbor_particles = cells[neighbor_key]

                if cell_key == neighbor_key:
                    for a in range(len(particle_list) - 1):
                        i = particle_list[a]
                        pos_i = self.positions[i]
                        for b in range(a + 1, len(particle_list)):
                            j = particle_list[b]
                            delta = self.positions[j] - pos_i
                            distance_sq = float(np.dot(delta, delta))

                            if distance_sq >= self.cutoff_sq or distance_sq < 1e-20:
                                continue

                            f_ij, u_ij, w_ij = self._lj_force_energy_virial(
                                delta,
                                distance_sq,
                            )
                            forces[i] -= f_ij
                            forces[j] += f_ij
                            total_potential += u_ij
                            total_virial += w_ij
                else:
                    for i in particle_list:
                        pos_i = self.positions[i]
                        for j in neighbor_particles:
                            delta = self.positions[j] - pos_i
                            distance_sq = float(np.dot(delta, delta))

                            if distance_sq >= self.cutoff_sq or distance_sq < 1e-20:
                                continue

                            f_ij, u_ij, w_ij = self._lj_force_energy_virial(
                                delta,
                                distance_sq,
                            )
                            forces[i] -= f_ij
                            forces[j] += f_ij
                            total_potential += u_ij
                            total_virial += w_ij

        return forces, total_potential, total_virial

    def reflect_at_sphere_boundary(self) -> None:
        effective_radius = self.radius - 0.05 * self.sigma

        for i in range(self.n_particles):
            position = self.positions[i]
            radius_sq = float(np.dot(position, position))

            if radius_sq > effective_radius**2:
                radius_value = np.sqrt(radius_sq)
                if radius_value < 1e-20:
                    continue

                normal = position / radius_value
                self.positions[i] = normal * effective_radius

                normal_velocity = float(np.dot(self.velocities[i], normal))
                if normal_velocity > 0:
                    self.velocities[i] -= (
                        (1.0 + self.params.wall_restitution)
                        * normal_velocity
                        * normal
                    )

    def step(self) -> None:
        dt = self.params.dt

        self.positions += (
            self.velocities * dt + 0.5 * (self.forces / self.mass) * dt * dt
        )

        self.reflect_at_sphere_boundary()

        new_forces, new_potential, new_virial = self.compute_forces()

        self.velocities += 0.5 * (self.forces + new_forces) * (dt / self.mass)

        self.reflect_at_sphere_boundary()

        self.forces = new_forces
        self.potential_energy = new_potential
        self.virial = new_virial

        self.time += dt
        self.step_index += 1
        self._log_state()

    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * float(np.sum(self.velocities * self.velocities))

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy

    def instantaneous_temperature(self) -> float:
        kinetic = self.kinetic_energy()
        dof = max(1, 3 * self.n_particles - 3)
        return 2.0 * kinetic / dof

    def pressure_estimate(self) -> float:
        volume = (4.0 / 3.0) * np.pi * self.radius**3
        temperature = self.instantaneous_temperature()
        return (self.n_particles * self.k_b * temperature + self.virial / 3.0) / volume

    def _log_state(self) -> None:
        kinetic = self.kinetic_energy()
        potential = self.potential_energy
        total = kinetic + potential
        temperature = self.instantaneous_temperature()
        pressure = self.pressure_estimate()

        self.energy_log.append((self.time, kinetic, potential, total))
        self.temperature_log.append((self.time, temperature))
        self.pressure_log.append((self.time, pressure))

        idx = int(np.clip(self.params.tracked_particle_index, 0, self.n_particles - 1))
        x, y, z = self.positions[idx]
        self.track_log.append((self.time, x, y, z))

    def save_csv_logs(self, base_path: str) -> tuple[str, str, str]:
        energy_path = f"{base_path}_energy.csv"
        track_path = f"{base_path}_track.csv"
        thermo_path = f"{base_path}_thermo.csv"

        stride = max(1, self.params.save_every)

        with open(energy_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "kinetic", "potential", "total"])
            for row in self.energy_log[::stride]:
                writer.writerow(row)

        with open(track_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "x", "y", "z"])
            for row in self.track_log[::stride]:
                writer.writerow(row)

        with open(thermo_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "temperature", "pressure_estimate"])
            for i in range(0, len(self.temperature_log), stride):
                time_value, temperature = self.temperature_log[i]
                _, pressure = self.pressure_log[i]
                writer.writerow([time_value, temperature, pressure])

        return energy_path, track_path, thermo_path


class SimulationViewer3D:
    def __init__(self, simulation: MolecularDynamicsLJ) -> None:
        self.simulation = simulation

        self.figure = plt.figure(figsize=(10, 8))
        self.axis = self.figure.add_subplot(111, projection="3d")

        r = self.simulation.radius
        self.axis.set_xlim(-r, r)
        self.axis.set_ylim(-r, r)
        self.axis.set_zlim(-r, r)
        self.axis.set_xlabel("x")
        self.axis.set_ylabel("y")
        self.axis.set_zlabel("z")
        self.axis.set_title("Molecular Dynamics in Sphere: Lennard-Jones + Velocity Verlet")

        self._draw_sphere_wireframe(r)

        self.scatter_particles = self.axis.scatter([], [], [], s=6, alpha=0.75)
        self.scatter_tracked = self.axis.scatter([], [], [], s=40)

        (self.track_line,) = self.axis.plot([], [], [], linewidth=1.2)
        self.track_points: list[np.ndarray] = []

        self.info_text = self.axis.text2D(
            0.02,
            0.98,
            "",
            transform=self.axis.transAxes,
            va="top",
        )

        self.animation: FuncAnimation | None = None

    def _draw_sphere_wireframe(self, radius: float) -> None:
        u = np.linspace(0, 2 * np.pi, 32)
        v = np.linspace(0, np.pi, 16)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones_like(u), np.cos(v))
        self.axis.plot_wireframe(x, y, z, linewidth=0.25, alpha=0.2)

    def _update(self, _frame: int):
        for _ in range(self.simulation.params.steps_per_frame):
            self.simulation.step()

        positions = self.simulation.positions
        self.scatter_particles._offsets3d = (
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
        )

        idx = int(
            np.clip(
                self.simulation.params.tracked_particle_index,
                0,
                self.simulation.n_particles - 1,
            )
        )
        tracked_position = self.simulation.positions[idx]
        self.scatter_tracked._offsets3d = (
            np.array([tracked_position[0]]),
            np.array([tracked_position[1]]),
            np.array([tracked_position[2]]),
        )

        self.track_points.append(tracked_position.copy())
        if len(self.track_points) > self.simulation.params.max_trajectory_points:
            self.track_points.pop(0)

        if len(self.track_points) > 1:
            trajectory = np.array(self.track_points)
            self.track_line.set_data(trajectory[:, 0], trajectory[:, 1])
            self.track_line.set_3d_properties(trajectory[:, 2])

        kinetic = self.simulation.kinetic_energy()
        potential = self.simulation.potential_energy
        total = kinetic + potential
        temperature = self.simulation.instantaneous_temperature()
        pressure = self.simulation.pressure_estimate()

        self.info_text.set_text(
            f"N = {self.simulation.n_particles}\n"
            f"t = {self.simulation.time:.4f}\n"
            f"dt = {self.simulation.params.dt}\n"
            f"K = {kinetic:.4f}\n"
            f"U = {potential:.4f}\n"
            f"E = {total:.4f}\n"
            f"T = {temperature:.4f}\n"
            f"P = {pressure:.4f}"
        )

        return (
            self.scatter_particles,
            self.scatter_tracked,
            self.track_line,
            self.info_text,
        )

    def run(self) -> None:
        self.animation = FuncAnimation(
            self.figure,
            self._update,
            interval=30,
            blit=False,
        )
        plt.tight_layout()
        plt.show()


class DiagnosticsPlotter:
    @staticmethod
    def show(simulation: MolecularDynamicsLJ) -> None:
        energy = np.array(simulation.energy_log)
        temperature = np.array(simulation.temperature_log)
        pressure = np.array(simulation.pressure_log)

        fig_energy = plt.figure(figsize=(8, 5))
        ax_energy = fig_energy.add_subplot(111)
        ax_energy.plot(energy[:, 0], energy[:, 1], label="Kinetic")
        ax_energy.plot(energy[:, 0], energy[:, 2], label="Potential")
        ax_energy.plot(energy[:, 0], energy[:, 3], label="Total")
        ax_energy.set_xlabel("t")
        ax_energy.set_ylabel("Energy")
        ax_energy.set_title("System Energy")
        ax_energy.grid(True, alpha=0.3)
        ax_energy.legend()

        fig_thermo = plt.figure(figsize=(8, 5))
        ax_thermo = fig_thermo.add_subplot(111)
        ax_thermo.plot(temperature[:, 0], temperature[:, 1], label="Temperature")
        ax_thermo.plot(pressure[:, 0], pressure[:, 1], label="Pressure estimate")
        ax_thermo.set_xlabel("t")
        ax_thermo.set_ylabel("Value")
        ax_thermo.set_title("Thermodynamic Diagnostics")
        ax_thermo.grid(True, alpha=0.3)
        ax_thermo.legend()

        plt.show()


class MdSimulationApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("MD in Sphere: Lennard-Jones + Maxwell-Boltzmann")

        self.entries: dict[str, ttk.Entry] = {}
        self.show_diagnostics_var = tk.BooleanVar(value=True)
        self.save_csv_var = tk.BooleanVar(value=False)

        self.simulation: MolecularDynamicsLJ | None = None

        self._build_ui()

    def _add_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        default_value: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)

        entry = ttk.Entry(parent, width=20)
        entry.grid(row=row, column=1, sticky="ew", pady=3, padx=(8, 0))
        entry.insert(0, default_value)

        self.entries[label] = entry

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        defaults = SimulationParameters()

        fields = [
            ("N (number of particles)", str(defaults.n_particles)),
            ("Sphere radius R", str(defaults.sphere_radius)),
            ("Time step dt", str(defaults.dt)),
            ("Steps per frame", str(defaults.steps_per_frame)),
            ("Particle mass", str(defaults.particle_mass)),
            ("Temperature T", str(defaults.temperature)),
            ("LJ epsilon", str(defaults.epsilon)),
            ("LJ sigma", str(defaults.sigma)),
            ("Cutoff (in sigma)", str(defaults.cutoff_sigma)),
            ("Init min distance (in sigma)", str(defaults.init_min_distance_sigma)),
            ("Tracked particle index", str(defaults.tracked_particle_index)),
            ("Save every Nth point", str(defaults.save_every)),
            ("Random seed (blank = random)", ""),
        ]

        row = 0
        for label, default in fields:
            self._add_entry(frame, row, label, default)
            row += 1

        ttk.Checkbutton(
            frame,
            text="Show diagnostics after animation",
            variable=self.show_diagnostics_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 2))
        row += 1

        ttk.Checkbutton(
            frame,
            text="Save CSV logs after animation",
            variable=self.save_csv_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=2, sticky="e", pady=(10, 0))

        ttk.Button(buttons, text="Start", command=self.start).grid(
            row=0,
            column=0,
            padx=4,
        )
        ttk.Button(buttons, text="Exit", command=self.root.destroy).grid(
            row=0,
            column=1,
            padx=4,
        )

    def _read_parameters(self) -> SimulationParameters:
        seed_text = self.entries["Random seed (blank = random)"].get().strip()
        seed = None if seed_text == "" else int(seed_text)

        params = SimulationParameters(
            n_particles=int(self.entries["N (number of particles)"].get()),
            sphere_radius=float(self.entries["Sphere radius R"].get()),
            dt=float(self.entries["Time step dt"].get()),
            steps_per_frame=int(self.entries["Steps per frame"].get()),
            particle_mass=float(self.entries["Particle mass"].get()),
            temperature=float(self.entries["Temperature T"].get()),
            epsilon=float(self.entries["LJ epsilon"].get()),
            sigma=float(self.entries["LJ sigma"].get()),
            cutoff_sigma=float(self.entries["Cutoff (in sigma)"].get()),
            init_min_distance_sigma=float(
                self.entries["Init min distance (in sigma)"].get()
            ),
            tracked_particle_index=int(self.entries["Tracked particle index"].get()),
            save_every=int(self.entries["Save every Nth point"].get()),
            seed=seed,
        )

        self._validate_parameters(params)
        return params

    @staticmethod
    def _validate_parameters(params: SimulationParameters) -> None:
        if params.n_particles < 2:
            raise ValueError("Number of particles must be at least 2.")

        if params.sphere_radius <= 0 or params.dt <= 0:
            raise ValueError("Sphere radius and dt must be positive.")

        if (
            params.particle_mass <= 0
            or params.temperature <= 0
            or params.epsilon <= 0
            or params.sigma <= 0
        ):
            raise ValueError("Mass, temperature, epsilon, and sigma must be positive.")

        if params.cutoff_sigma <= 1.0:
            raise ValueError("Cutoff (in sigma) should be greater than 1.0.")

        if params.steps_per_frame < 1:
            raise ValueError("Steps per frame must be at least 1.")

        if params.save_every < 1:
            raise ValueError("Save every Nth point must be at least 1.")

    def start(self) -> None:
        try:
            params = self._read_parameters()
        except Exception as error:
            messagebox.showerror("Input Error", str(error))
            return

        try:
            self.root.withdraw()

            self.simulation = MolecularDynamicsLJ(params)
            viewer = SimulationViewer3D(self.simulation)
            viewer.run()

            if self.show_diagnostics_var.get():
                DiagnosticsPlotter.show(self.simulation)

            if self.save_csv_var.get():
                base_path = filedialog.asksaveasfilename(
                    title="Save CSV logs (choose base file name)",
                    defaultextension="",
                    filetypes=[("All files", "*.*")],
                )
                if base_path:
                    energy_path, track_path, thermo_path = self.simulation.save_csv_logs(
                        base_path
                    )
                    messagebox.showinfo(
                        "Saved",
                        f"Files saved:\n{energy_path}\n{track_path}\n{thermo_path}",
                    )

        except Exception as error:
            messagebox.showerror("Simulation Error", str(error))
        finally:
            self.root.deiconify()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    MdSimulationApp().run()
