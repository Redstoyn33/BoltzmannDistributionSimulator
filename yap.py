import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from dataclasses import dataclass

# =========================
# МОЛЕКУЛЯРНАЯ ДИНАМИКА В СФЕРЕ (LJ + Velocity Verlet)
# =========================

@dataclass
class MDParams:
    n_particles: int = 800
    sphere_radius: float = 12.0
    dt: float = 0.002
    steps_per_frame: int = 2
    mass: float = 1.0
    temperature: float = 1.0
    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff: float = 2.5
    wall_kick_damping: float = 1.0  # 1.0 = абсолютно упруго
    seed: int | None = None
    init_min_dist: float = 0.9
    track_index: int = 0
    save_every: int = 10
    max_traj_points: int = 500


class LJMDInSphere:
    def __init__(self, p: MDParams):
        self.p = p
        self.rng = np.random.default_rng(p.seed)

        # Приведенные единицы: k_B = 1
        self.kB = 1.0

        # Центр сферы в начале координат
        self.R = float(p.sphere_radius)
        self.N = int(p.n_particles)
        self.dim = 3

        self.m = np.full(self.N, float(p.mass), dtype=float)
        self.pos = np.zeros((self.N, 3), dtype=float)
        self.vel = np.zeros((self.N, 3), dtype=float)
        self.force = np.zeros((self.N, 3), dtype=float)

        self.time = 0.0
        self.step_count = 0

        # Параметры LJ
        self.eps = float(p.epsilon)
        self.sig = float(p.sigma)
        self.rc = float(p.cutoff) * self.sig
        self.rc2 = self.rc * self.rc

        # Сдвиг потенциала на отсечении, чтобы U(rc)=0
        sr6c = (self.sig / self.rc) ** 6
        self.U_shift = 4.0 * self.eps * (sr6c * sr6c - sr6c)

        # Инициализация
        self._init_positions_in_sphere(min_dist=p.init_min_dist * self.sig)
        self._init_velocities_MB(T=p.temperature)

        # Начальные силы
        self.potential_energy = 0.0
        self.virial = 0.0
        self.force, self.potential_energy, self.virial = self.compute_forces_cell_list()

        # Журналы
        self.energy_log = []
        self.temp_log = []
        self.pressure_like_log = []
        self.track_log = []

        self._log_state()

    # -------------------------
    # ИНИЦИАЛИЗАЦИЯ
    # -------------------------
    def _random_point_in_sphere(self, margin=0.0):
        # Равномерно по объему сферы
        while True:
            x = self.rng.uniform(-(self.R - margin), self.R - margin, size=3)
            if np.dot(x, x) <= (self.R - margin) ** 2:
                return x

    def _init_positions_in_sphere(self, min_dist: float):
        # Простая случайная упаковка без сильных перекрытий
        # Для тысяч частиц нужна разумная плотность, иначе размещение будет медленным.
        min_dist2 = min_dist * min_dist
        for i in range(self.N):
            placed = False
            for _ in range(20000):
                cand = self._random_point_in_sphere(margin=0.2 * self.sig)
                if i == 0:
                    cand = np.array([0.0, 0.0, 0.0], dtype=float)
                ok = True
                for j in range(i):
                    d = cand - self.pos[j]
                    if np.dot(d, d) < min_dist2:
                        ok = False
                        break
                if ok:
                    self.pos[i] = cand
                    placed = True
                    break
            if not placed:
                raise RuntimeError(
                    "Не удалось разместить частицы без сильных перекрытий. "
                    "Уменьшите число частиц, увеличьте радиус сферы или уменьшите init_min_dist."
                )

    def _init_velocities_MB(self, T: float):
        # Максвелл-Больцман: компоненты скорости гауссовы, sigma_v = sqrt(k_B T / m)
        sigma_v = np.sqrt(self.kB * T / self.p.mass)
        self.vel = self.rng.normal(0.0, sigma_v, size=(self.N, 3))

        # Убираем суммарный импульс, чтобы центр масс не дрейфовал
        v_cm = np.mean(self.vel, axis=0)
        self.vel -= v_cm

        # Точно подгоняем температуру: T = (2K)/(3N-3) при нулевом импульсе
        K = self.kinetic_energy()
        dof = 3 * self.N - 3
        if dof > 0 and K > 0:
            T_current = 2.0 * K / dof
            scale = np.sqrt(T / T_current)
            self.vel *= scale

    # -------------------------
    # СИЛЫ ЛЕННАРДА-ДЖОНСА + CELL LIST
    # -------------------------
    def compute_forces_cell_list(self):
        forces = np.zeros_like(self.pos)
        U = 0.0
        virial = 0.0

        # Куб, описанный вокруг сферы: [-R, R]^3
        box_min = -self.R
        box_max = self.R

        # Размер ячейки берем не меньше радиуса отсечения
        cell_size = self.rc
        ncell = max(1, int(np.floor((box_max - box_min) / cell_size)))
        cell_size = (box_max - box_min) / ncell

        # Раскладка частиц по ячейкам
        cells = {}
        idx_all = np.arange(self.N)

        scaled = (self.pos - box_min) / cell_size
        cid = np.floor(scaled).astype(int)
        cid = np.clip(cid, 0, ncell - 1)

        for i in idx_all:
            key = (cid[i, 0], cid[i, 1], cid[i, 2])
            if key not in cells:
                cells[key] = []
            cells[key].append(i)

        # Соседние ячейки
        neigh_shifts = [(dx, dy, dz)
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                        for dz in (-1, 0, 1)]

        # Перебор пар без двойного счета
        visited_pairs_cells = set()

        for cell_key, plist in cells.items():
            cx, cy, cz = cell_key
            for sh in neigh_shifts:
                nk = (cx + sh[0], cy + sh[1], cz + sh[2])
                if nk not in cells:
                    continue

                # Чтобы не считать дважды пары ячеек
                pair_cell = tuple(sorted((cell_key, nk)))
                if pair_cell in visited_pairs_cells:
                    continue
                visited_pairs_cells.add(pair_cell)

                qlist = cells[nk]

                if cell_key == nk:
                    # Пары внутри одной ячейки
                    for a in range(len(plist) - 1):
                        i = plist[a]
                        ri = self.pos[i]
                        for b in range(a + 1, len(plist)):
                            j = plist[b]
                            rij = self.pos[j] - ri
                            r2 = np.dot(rij, rij)
                            if r2 >= self.rc2 or r2 < 1e-20:
                                continue
                            fij, uij, wij = self._lj_pair_force_energy_virial(rij, r2)
                            forces[i] -= fij
                            forces[j] += fij
                            U += uij
                            virial += wij
                else:
                    # Пары между разными ячейками
                    for i in plist:
                        ri = self.pos[i]
                        for j in qlist:
                            rij = self.pos[j] - ri
                            r2 = np.dot(rij, rij)
                            if r2 >= self.rc2 or r2 < 1e-20:
                                continue
                            fij, uij, wij = self._lj_pair_force_energy_virial(rij, r2)
                            forces[i] -= fij
                            forces[j] += fij
                            U += uij
                            virial += wij

        return forces, U, virial

    def _lj_pair_force_energy_virial(self, rij, r2):
        # LJ в векторной форме
        # U = 4e[(s/r)^12 - (s/r)^6] - U(rc)
        # F_vec = 24e[2(s^12/r^14) - (s^6/r^8)] * rij
        inv_r2 = 1.0 / r2
        sr2 = (self.sig * self.sig) * inv_r2
        sr6 = sr2 ** 3
        sr12 = sr6 ** 2

        # Потенциальная энергия пары со сдвигом
        u = 4.0 * self.eps * (sr12 - sr6) - self.U_shift

        # Векторная сила
        coef = 24.0 * self.eps * (2.0 * sr12 - sr6) * inv_r2
        fvec = coef * rij

        # Вириал пары для оценки давления: w = r_ij · F_ij
        wij = np.dot(rij, fvec)

        return fvec, u, wij

    # -------------------------
    # ГРАНИЦА СФЕРЫ
    # -------------------------
    def _reflect_from_sphere(self):
        # Если частица вышла за сферу, отражаем скорость по нормали и возвращаем на границу
        R_eff = self.R - 0.05 * self.sig
        for i in range(self.N):
            r = self.pos[i]
            rr = np.dot(r, r)
            if rr > R_eff * R_eff:
                norm = np.sqrt(rr)
                if norm < 1e-20:
                    continue
                n = r / norm

                # Перенос на поверхность чуть внутрь
                self.pos[i] = n * R_eff

                # Отражение нормальной компоненты скорости
                vn = np.dot(self.vel[i], n)
                if vn > 0:
                    self.vel[i] = self.vel[i] - (1.0 + self.p.wall_kick_damping) * vn * n

    # -------------------------
    # ИНТЕГРИРОВАНИЕ (VELOCITY VERLET)
    # -------------------------
    def step(self):
        dt = self.p.dt
        m = self.p.mass

        # x(t+dt) = x(t) + v(t)dt + (1/2m)F(t)dt^2
        self.pos += self.vel * dt + 0.5 * (self.force / m) * dt * dt

        # Граница сферы после шага координат
        self._reflect_from_sphere()

        # Новые силы
        new_force, U_new, vir_new = self.compute_forces_cell_list()

        # v(t+dt) = v(t) + (1/2m)(F(t)+F(t+dt))dt
        self.vel += 0.5 * (self.force + new_force) * (dt / m)

        # Повторная коррекция для частиц, ушедших после обновления скоростей
        self._reflect_from_sphere()

        self.force = new_force
        self.potential_energy = U_new
        self.virial = vir_new

        self.time += dt
        self.step_count += 1
        self._log_state()

    # -------------------------
    # НАБЛЮДАЕМЫЕ ВЕЛИЧИНЫ
    # -------------------------
    def kinetic_energy(self):
        return 0.5 * self.p.mass * np.sum(self.vel * self.vel)

    def total_energy(self):
        return self.kinetic_energy() + self.potential_energy

    def instantaneous_temperature(self):
        # При удаленном движении центра масс число степеней свободы примерно 3N-3
        K = self.kinetic_energy()
        dof = max(1, 3 * self.N - 3)
        return 2.0 * K / dof

    def pressure_like(self):
        # Оценка через вириальную формулу для сферы объема V = 4/3 pi R^3
        # P = (N k_B T + (1/3) <sum r_ij·F_ij>/2? ) / V
        # Здесь virial уже суммируется по парам один раз, поэтому используем P = (N T + virial/3) / V
        V = 4.0 * np.pi * (self.R ** 3) / 3.0
        T = self.instantaneous_temperature()
        return (self.N * self.kB * T + self.virial / 3.0) / V

    def _log_state(self):
        T = self.instantaneous_temperature()
        self.energy_log.append((self.time, self.kinetic_energy(), self.potential_energy, self.total_energy()))
        self.temp_log.append((self.time, T))
        self.pressure_like_log.append((self.time, self.pressure_like()))

        idx = int(np.clip(self.p.track_index, 0, self.N - 1))
        self.track_log.append((self.time, self.pos[idx, 0], self.pos[idx, 1], self.pos[idx, 2]))

    # -------------------------
    # ВЫХОДНЫЕ ДАННЫЕ
    # -------------------------
    def save_logs_csv(self, base_path):
        energy_path = base_path + "_energy.csv"
        track_path = base_path + "_track.csv"
        temp_path = base_path + "_temp_pressure.csv"

        with open(energy_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "K", "U", "E"])
            for row in self.energy_log[::max(1, self.p.save_every)]:
                w.writerow(row)

        with open(track_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "x", "y", "z"])
            for row in self.track_log[::max(1, self.p.save_every)]:
                w.writerow(row)

        with open(temp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "T", "P_like"])
            for i in range(0, len(self.temp_log), max(1, self.p.save_every)):
                t, T = self.temp_log[i]
                _, P = self.pressure_like_log[i]
                w.writerow([t, T, P])

        return energy_path, track_path, temp_path


# =========================
# ВИЗУАЛИЗАЦИЯ
# =========================

class MDViewer:
    def __init__(self, sim: LJMDInSphere):
        self.sim = sim

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        R = sim.R
        self.ax.set_xlim(-R, R)
        self.ax.set_ylim(-R, R)
        self.ax.set_zlim(-R, R)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_title("Молекулярная динамика в сфере: Lennard-Jones + Velocity Verlet")

        # Сфера (каркас)
        self._draw_sphere_wireframe(R)

        # Частицы
        self.scat = self.ax.scatter([], [], [], s=6, alpha=0.75)
        idx = int(np.clip(self.sim.p.track_index, 0, self.sim.N - 1))
        self.scat_track = self.ax.scatter([], [], [], s=40)

        # Траектория отслеживаемой частицы
        self.line_track, = self.ax.plot([], [], [], linewidth=1.2)

        self.info = self.ax.text2D(0.02, 0.98, "", transform=self.ax.transAxes, va="top")

        self.max_track = self.sim.p.max_traj_points
        self.track_pts = []

    def _draw_sphere_wireframe(self, R):
        u = np.linspace(0, 2 * np.pi, 32)
        v = np.linspace(0, np.pi, 16)
        x = R * np.outer(np.cos(u), np.sin(v))
        y = R * np.outer(np.sin(u), np.sin(v))
        z = R * np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_wireframe(x, y, z, linewidth=0.25, alpha=0.2)

    def _update(self, frame):
        for _ in range(self.sim.p.steps_per_frame):
            self.sim.step()

        P = self.sim.pos
        self.scat._offsets3d = (P[:, 0], P[:, 1], P[:, 2])

        idx = int(np.clip(self.sim.p.track_index, 0, self.sim.N - 1))
        r = self.sim.pos[idx]
        self.scat_track._offsets3d = (np.array([r[0]]), np.array([r[1]]), np.array([r[2]]))

        self.track_pts.append(r.copy())
        if len(self.track_pts) > self.max_track:
            self.track_pts.pop(0)
        tr = np.array(self.track_pts)
        if len(tr) > 1:
            self.line_track.set_data(tr[:, 0], tr[:, 1])
            self.line_track.set_3d_properties(tr[:, 2])

        K = self.sim.kinetic_energy()
        U = self.sim.potential_energy
        E = K + U
        T = self.sim.instantaneous_temperature()
        P_like = self.sim.pressure_like()

        self.info.set_text(
            f"N = {self.sim.N}\n"
            f"t = {self.sim.time:.4f}\n"
            f"dt = {self.sim.p.dt}\n"
            f"K = {K:.4f}\n"
            f"U = {U:.4f}\n"
            f"E = {E:.4f}\n"
            f"T = {T:.4f}\n"
            f"P = {P_like:.4f}"
        )
        return self.scat, self.scat_track, self.line_track, self.info

    def run(self):
        self.anim = FuncAnimation(self.fig, self._update, interval=30, blit=False)
        plt.tight_layout()
        plt.show()


# =========================
# ГРАФИКИ ПОСЛЕ СИМУЛЯЦИИ
# =========================

def show_diagnostics(sim: LJMDInSphere):
    e = np.array(sim.energy_log)
    tp = np.array(sim.temp_log)
    pp = np.array(sim.pressure_like_log)

    fig1 = plt.figure(figsize=(8, 5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(e[:, 0], e[:, 1], label="K")
    ax1.plot(e[:, 0], e[:, 2], label="U")
    ax1.plot(e[:, 0], e[:, 3], label="E")
    ax1.set_xlabel("t")
    ax1.set_ylabel("energy")
    ax1.set_title("Энергия системы")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig2 = plt.figure(figsize=(8, 5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(tp[:, 0], tp[:, 1], label="T")
    ax2.plot(pp[:, 0], pp[:, 1], label="P_like")
    ax2.set_xlabel("t")
    ax2.set_ylabel("value")
    ax2.set_title("Температура и вириальная оценка давления")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.show()


# =========================
# GUI: ВВОД И ВЫВОД
# =========================

class MDApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MD в сфере: Lennard-Jones + Максвелл-Больцман")

        self.entries = {}
        self._build_ui()
        self.sim = None

    def _add_field(self, parent, row, label, default):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        e = ttk.Entry(parent, width=18)
        e.grid(row=row, column=1, sticky="ew", pady=3, padx=(8, 0))
        e.insert(0, str(default))
        self.entries[label] = e

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)

        defaults = MDParams()

        row = 0
        self._add_field(frm, row, "N (число частиц)", defaults.n_particles); row += 1
        self._add_field(frm, row, "R (радиус сферы)", defaults.sphere_radius); row += 1
        self._add_field(frm, row, "dt", defaults.dt); row += 1
        self._add_field(frm, row, "steps_per_frame", defaults.steps_per_frame); row += 1
        self._add_field(frm, row, "mass", defaults.mass); row += 1
        self._add_field(frm, row, "T (температура)", defaults.temperature); row += 1
        self._add_field(frm, row, "epsilon", defaults.epsilon); row += 1
        self._add_field(frm, row, "sigma", defaults.sigma); row += 1
        self._add_field(frm, row, "cutoff (в sigma)", defaults.cutoff); row += 1
        self._add_field(frm, row, "init_min_dist (в sigma)", defaults.init_min_dist); row += 1
        self._add_field(frm, row, "track_index", defaults.track_index); row += 1
        self._add_field(frm, row, "save_every", defaults.save_every); row += 1
        self._add_field(frm, row, "seed (пусто=случайно)", ""); row += 1

        self.auto_diag_var = tk.BooleanVar(value=True)
        self.save_csv_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(frm, text="Показать графики после визуализации", variable=self.auto_diag_var)\
            .grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 2)); row += 1
        ttk.Checkbutton(frm, text="Сохранить CSV после закрытия визуализации", variable=self.save_csv_var)\
            .grid(row=row, column=0, columnspan=2, sticky="w", pady=2); row += 1

        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=2, sticky="e", pady=(10, 0))

        ttk.Button(btns, text="Старт", command=self.run_sim).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Выход", command=self.root.destroy).grid(row=0, column=1, padx=4)

    def _read_params(self):
        try:
            seed_text = self.entries["seed (пусто=случайно)"].get().strip()
            seed = None if seed_text == "" else int(seed_text)

            p = MDParams(
                n_particles=int(self.entries["N (число частиц)"].get()),
                sphere_radius=float(self.entries["R (радиус сферы)"].get()),
                dt=float(self.entries["dt"].get()),
                steps_per_frame=int(self.entries["steps_per_frame"].get()),
                mass=float(self.entries["mass"].get()),
                temperature=float(self.entries["T (температура)"].get()),
                epsilon=float(self.entries["epsilon"].get()),
                sigma=float(self.entries["sigma"].get()),
                cutoff=float(self.entries["cutoff (в sigma)"].get()),
                init_min_dist=float(self.entries["init_min_dist (в sigma)"].get()),
                track_index=int(self.entries["track_index"].get()),
                save_every=int(self.entries["save_every"].get()),
                seed=seed
            )

            if p.n_particles < 2:
                raise ValueError("N должно быть не меньше 2")
            if p.sphere_radius <= 0 or p.dt <= 0:
                raise ValueError("R и dt должны быть положительными")
            if p.mass <= 0 or p.temperature <= 0 or p.epsilon <= 0 or p.sigma <= 0:
                raise ValueError("mass, T, epsilon, sigma должны быть положительными")
            if p.cutoff <= 1.0:
                raise ValueError("cutoff должен быть больше 1 (обычно 2.5)")
            if p.steps_per_frame < 1:
                raise ValueError("steps_per_frame должно быть не меньше 1")

            return p

        except Exception as ex:
            messagebox.showerror("Ошибка ввода", str(ex))
            return None

    def run_sim(self):
        p = self._read_params()
        if p is None:
            return

        try:
            self.root.withdraw()

            self.sim = LJMDInSphere(p)
            viewer = MDViewer(self.sim)
            viewer.run()

            if self.auto_diag_var.get():
                show_diagnostics(self.sim)

            if self.save_csv_var.get():
                base = filedialog.asksaveasfilename(
                    title="Сохранить CSV (укажите базовое имя)",
                    defaultextension="",
                    filetypes=[("All files", "*.*")]
                )
                if base:
                    epath, tpath, ppath = self.sim.save_logs_csv(base)
                    messagebox.showinfo(
                        "Сохранено",
                        f"Файлы сохранены:\n{epath}\n{tpath}\n{ppath}"
                    )

        except Exception as ex:
            messagebox.showerror("Ошибка симуляции", str(ex))
        finally:
            self.root.deiconify()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    MDApp().run()