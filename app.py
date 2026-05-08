import streamlit as st
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from scipy.ndimage import gaussian_filter
import math

# ---------------------------------------------------------
# Caminhos
# ---------------------------------------------------------

import os

BASE_DIR = os.path.dirname(__file__)

DATA_JSON = os.path.join(BASE_DIR, "data", "0_data.json")
MATCHES_DIR = os.path.join(BASE_DIR, "data")

MINIMAP_LANE = os.path.join(BASE_DIR, "assets", "simpleMinimap_7.40_400.png")
MINIMAP_WARDS = os.path.join(BASE_DIR, "assets", "simpleMinimap_7.40.png")

SMOKE_AGG_PATH = os.path.join(BASE_DIR, "replay", "smoke_events_aggregated.json")
SMOKE_MAPImage = MINIMAP_WARDS

IMG_LANE = 400
IMG_WARDS = 550

RADIANT_SLOTS = {0, 1, 2, 3, 4}
DIRE_SLOTS = {128, 129, 130, 131, 132}

# ---------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------

def load_data_json():
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def load_match(filename):
    path = os.path.join(MATCHES_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# LANE/DEATH → 400x400
# ---------------------------------------------------------

def get_lane_bounds(data):
    xs, ys = [], []
    for p in data["players"]:
        lp = p.get("lane_pos", {})
        for x_str, ys_dict in lp.items():
            xs.append(int(x_str))
            for y_str in ys_dict.keys():
                ys.append(int(y_str))
    return min(xs), max(xs), min(ys), max(ys)

def lane_to_pixel_base(x, y, x_min, x_max, y_min, y_max):
    px_raw = (x - x_min) / (x_max - x_min) * (IMG_LANE - 1)
    py_norm = (y - y_min) / (y_max - y_min)
    py_raw = (1 - py_norm) * (IMG_LANE - 1)
    return px_raw, py_raw

def adjust_to_opendota(px, py):
    scale = 1.009
    px = px * scale + 9
    py = py * scale - 32
    return int(px), int(py)

def lane_to_pixel_400(x, y, x_min, x_max, y_min, y_max):
    px_raw, py_raw = lane_to_pixel_base(x, y, x_min, x_max, y_min, y_max)
    return adjust_to_opendota(px_raw, py_raw)


# ---------------------------------------------------------
# DEATHS → 550x550 (grid 0–160 → 400 → 550)
# ---------------------------------------------------------

def death_to_pixel_550(x, y, x_min, x_max, y_min, y_max):
    px400_raw, py400_raw = lane_to_pixel_base(x, y, x_min, x_max, y_min, y_max)
    px400, py400 = adjust_to_opendota(px400_raw, py400_raw)

    scale = IMG_WARDS / IMG_LANE
    return int(px400 * scale), int(py400 * scale)


# ---------------------------------------------------------
# WARDS → 550x550 (raw → OpenDota)
# ---------------------------------------------------------

A_X = 4.348
B_X = -279
A_Y = -4.324
B_Y = 827

def raw_to_od(x_raw, y_raw):
    px = A_X * x_raw + B_X
    py = A_Y * y_raw + B_Y
    return int(round(px)), int(round(py))


# ---------------------------------------------------------
# LANE POSITION HEATMAP (400x400)
# ---------------------------------------------------------

def aggregate_lane_pos(matches, player_name):
    heat = np.zeros((IMG_LANE, IMG_LANE))

    for fname in matches:
        data = load_match(fname)
        x_min, x_max, y_min, y_max = get_lane_bounds(data)

        for p in data["players"]:
            if p.get("name") != player_name:
                continue

            lp = p.get("lane_pos", {})
            for x_str, ys in lp.items():
                x = int(x_str)
                for y_str, count in ys.items():
                    y = int(y_str)
                    px, py = lane_to_pixel_400(x, y, x_min, x_max, y_min, y_max)
                    if 0 <= px < IMG_LANE and 0 <= py < IMG_LANE:
                        heat[py, px] += count

    return heat

# ---------------------------------------------------------
# DEATH POSITION HEATMAP (400x400)
# ---------------------------------------------------------

def aggregate_deaths_pos(matches, player_name):
    heat = np.zeros((IMG_LANE, IMG_LANE))

    for fname in matches:
        data = load_match(fname)
        x_min, x_max, y_min, y_max = get_lane_bounds(data)
        players = data["players"]

        for tf in data.get("teamfights", []):
            tf_players = tf.get("players", [])

            for i, tfp in enumerate(tf_players):
                if i >= len(players):
                    continue
                if players[i].get("name") != player_name:
                    continue

                deaths_pos = tfp.get("deaths_pos", {})
                for x_str, ys in deaths_pos.items():
                    x = int(x_str)
                    for y_str, count in ys.items():
                        y = int(y_str)
                        px, py = lane_to_pixel_400(x, y, x_min, x_max, y_min, y_max)
                        if 0 <= px < IMG_LANE and 0 <= py < IMG_LANE:
                            heat[py, px] += count

    return heat


# ---------------------------------------------------------
# WARDS (550x550)
# ---------------------------------------------------------

def extract_wards_from_match(data, side):
    wards = []

    for p in data["players"]:
        slot = p.get("player_slot")

        if side == "Radiant" and slot not in RADIANT_SLOTS:
            continue
        if side == "Dire" and slot not in DIRE_SLOTS:
            continue

        for obs in p.get("obs_log", []):
            wards.append({
                "type": "obs",
                "time": obs["time"],
                "x": obs["x"],
                "y": obs["y"],
                "ehandle": obs["ehandle"],
                "left_time": None
            })

        for sen in p.get("sen_log", []):
            wards.append({
                "type": "sen",
                "time": sen["time"],
                "x": sen["x"],
                "y": sen["y"],
                "ehandle": sen["ehandle"],
                "left_time": None
            })

        for left in p.get("obs_left_log", []):
            for w in wards:
                if w["ehandle"] == left["ehandle"]:
                    w["left_time"] = left["time"]

        for left in p.get("sen_left_log", []):
            for w in wards:
                if w["ehandle"] == left["ehandle"]:
                    w["left_time"] = left["time"]

    return wards


# ---------------------------------------------------------
# DEATHS FOR WARD VISION (550x550)
# ---------------------------------------------------------

def extract_all_deaths(matches, side):
    deaths = []

    for fname in matches:
        data = load_match(fname)
        players = data["players"]

        # bounds iguais ao lane_pos
        xs, ys = [], []
        for p in players:
            lp = p.get("lane_pos", {})
            for x_str, ys_dict in lp.items():
                xs.append(int(x_str))
                for y_str in ys_dict.keys():
                    ys.append(int(y_str))

        if not xs or not ys:
            continue

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        for tf in data.get("teamfights", []):
            tf_start = tf.get("start", 0)
            tf_end = tf.get("end", tf_start)

            tf_players = tf.get("players", [])

            for i, tfp in enumerate(tf_players):
                if i >= 10:
                    continue

                deaths_pos = tfp.get("deaths_pos", {})
                if not deaths_pos:
                    continue

                for x_str, ys in deaths_pos.items():
                    x = int(x_str)
                    for y_str, count in ys.items():
                        y = int(y_str)

                        px, py = death_to_pixel_550(x, y, x_min, x_max, y_min, y_max)

                        # QUEM MORREU:
                        # 0–4 = Radiant → VERDE
                        # 5–9 = Dire    → VERMELHO
                        color = "green" if i <= 4 else "red"

                        deaths.append({
                            "px": px,
                            "py": py,
                            "color": color,
                            "tf_start": tf_start,
                            "tf_end": tf_end
                        })

    return deaths


# ---------------------------------------------------------
# WARD VISION FILTER (550x550)
# ---------------------------------------------------------

def deaths_in_vision(deaths, wards):
    OBS_RADIUS = 69
    SEN_RADIUS = 43

    filtered = []

    for d in deaths:
        px_d, py_d = d["px"], d["py"]
        tf_start = d["tf_start"]
        tf_end = d["tf_end"]

        for w in wards:
            w_start = w["time"]
            if w["left_time"] is not None:
                w_end = w["left_time"]
            else:
                max_duration = 360 if w["type"] == "obs" else 420
                w_end = w_start + max_duration

            # precisa haver interseção entre [tf_start, tf_end] e [w_start, w_end]
            if tf_end < w_start or tf_start > w_end:
                continue

            px_w, py_w = raw_to_od(w["x"], w["y"])

            dx = px_d - px_w
            dy = py_d - py_w
            dist = math.sqrt(dx*dx + dy*dy)

            radius = OBS_RADIUS if w["type"] == "obs" else SEN_RADIUS

            if dist <= radius:
                filtered.append({
                    "px": px_d,
                    "py": py_d,
                    "color": d["color"]
                })
                break

    return filtered



# ---------------------------------------------------------
# PLOT WARDS + DEATHS (550x550)
# ---------------------------------------------------------

def plot_wards_and_deaths(wards, deaths, time_min, time_max):
    minimap = np.array(Image.open(MINIMAP_WARDS))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(minimap)

    # --- WARDS ---
    for w in wards:
        if not (time_min <= w["time"] <= time_max):
            continue

        px, py = raw_to_od(w["x"], w["y"])
        color = "#FFA500" if w["type"] == "obs" else "#00FF00"

        # raio da visão
        radius = 69 if w["type"] == "obs" else 43

        # círculo de visão (causava distorção antes)
        circle = Circle(
            (px, py),
            radius,
            edgecolor=color,
            facecolor=color,
            alpha=0.15
        )
        ax.add_patch(circle)

        # ponto central da ward
        ax.scatter(px, py, c=color, s=40, alpha=1.0, edgecolors="black")

        # tempo da ward (se expirou antes do limite)
        if w["left_time"] is not None:
            duration = w["left_time"] - w["time"]
            max_duration = 360 if w["type"] == "obs" else 420
            if duration < max_duration:
                ax.text(px, py - 10, f"{duration}s", color=color, fontsize=8, ha="center")

    # --- DEATHS ---
    for d in deaths:
        ax.scatter(
            d["px"],
            d["py"],
            c=d["color"],
            s=80,
            marker="s",
            edgecolors="black"
        )

    # --- CORREÇÃO CRÍTICA ---
    # Impede que os círculos alterem o tamanho do mapa
    ax.set_xlim(0, IMG_WARDS)
    ax.set_ylim(IMG_WARDS, 0)
    ax.set_aspect('equal', adjustable='box')

    ax.axis("off")
    return fig



# ---------------------------------------------------------
# HEATMAP PLOT (400x400)
# ---------------------------------------------------------

def plot_heatmap_400(heat):
    minimap = np.array(Image.open(MINIMAP_LANE))
    heat_log = np.log1p(heat)
    heat_gamma = heat_log ** 0.01
    heat_blur = gaussian_filter(heat_gamma, sigma=8)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(minimap)
    ax.imshow(heat_blur, cmap="hot", alpha=0.60)
    ax.axis("off")
    return fig

# ---------------------------------------------------------
# STREAMLIT PSWD
# ---------------------------------------------------------

import streamlit as st

st.set_page_config(page_title="Login", page_icon="🔒")

# Recupera a senha do secrets
PASSWORD = st.secrets["pswd"]

# Estado de autenticação
if "auth" not in st.session_state:
    st.session_state.auth = False

st.title("🔒 Área Restrita")

if not st.session_state.auth:
    pwd = st.text_input("Digite a senha:", type="password")

    if st.button("Entrar"):
        if pwd == PASSWORD:
            st.session_state.auth = True
            st.success("Acesso liberado!")
        else:
            st.error("Senha incorreta.")
else:
    st.success("Você está logado!")
    st.write("Conteúdo protegido aqui...")

    if st.button("Sair"):
        st.session_state.auth = False

    # ---------------------------------------------------------
    # STREAMLIT UI
    # ---------------------------------------------------------

    st.title("Dota 2 — Analyzer")

    data = load_data_json()

    tab_lane, tab_death, tab_wards, tab_smoke = st.tabs(["Lane Position", "Death Position", "Observer & Sentry", "8785561247"])


    # ---------------------------------------------------------
    # TAB 1 — LANE POSITION (400x400)
    # ---------------------------------------------------------

    with tab_lane:
        st.header("Lane Position Heatmap")

        teams = sorted(set(list(data["radiant"].keys()) + list(data["dire"].keys())))
        team = st.selectbox("Selecione o time:", teams, key="lane_team")

        side = st.radio("Selecione o lado:", ["Radiant", "Dire"], key="lane_side")

        if side == "Radiant":
            matches = data["radiant"][team]["matches"]
            players = data["radiant"][team]["players"]
        else:
            matches = data["dire"][team]["matches"]
            players = data["dire"][team]["players"]

        st.write(f"Partidas encontradas: **{len(matches)}**")

        player = st.selectbox("Selecione o jogador:", players, key="lane_player")

        if st.button("Gerar Lane Heatmap"):
            heat = aggregate_lane_pos(matches, player)
            fig = plot_heatmap_400(heat)
            st.pyplot(fig)


    # ---------------------------------------------------------
    # TAB 2 — DEATH POSITION (400x400)
    # ---------------------------------------------------------

    with tab_death:
        st.header("Death Position Heatmap")

        teams = sorted(set(list(data["radiant"].keys()) + list(data["dire"].keys())))
        team = st.selectbox("Selecione o time:", teams, key="death_team")

        side = st.radio("Selecione o lado:", ["Radiant", "Dire"], key="death_side")

        if side == "Radiant":
            matches = data["radiant"][team]["matches"]
            players = data["radiant"][team]["players"]
        else:
            matches = data["dire"][team]["matches"]
            players = data["dire"][team]["players"]

        st.write(f"Partidas encontradas: **{len(matches)}**")

        player = st.selectbox("Selecione o jogador:", players, key="death_player")

        if st.button("Gerar Death Heatmap"):
            heat = aggregate_deaths_pos(matches, player)
            fig = plot_heatmap_400(heat)
            st.pyplot(fig)

    # ---------------------------------------------------------
    # Função necessária para extrair mortes por TF
    # ---------------------------------------------------------

    def extract_deaths_from_teamfight(tf):
        deaths = []

        tf_start = tf["start"]
        tf_end = tf["end"]

        for slot, tfp in enumerate(tf["players"]):
            deaths_pos = tfp.get("deaths_pos", {})
            if not deaths_pos:
                continue

            for x_str, ys in deaths_pos.items():
                x = int(x_str)
                for y_str, count in ys.items():
                    y = int(y_str)

                    deaths.append({
                        "time": tf_start,
                        "slot": slot,
                        "x": x,
                        "y": y,
                        "tf_start": tf_start,
                        "tf_end": tf_end
                    })

        return deaths

    # ---------------------------------------------------------
    # Função necessária para extrair mortes por TF
    # ---------------------------------------------------------

    def extract_deaths_from_teamfight(tf):
        deaths = []

        tf_start = tf["start"]
        tf_end = tf["end"]

        for slot, tfp in enumerate(tf["players"]):
            deaths_pos = tfp.get("deaths_pos", {})
            if not deaths_pos:
                continue

            for x_str, ys in deaths_pos.items():
                x = int(x_str)
                for y_str, count in ys.items():
                    y = int(y_str)

                    deaths.append({
                        "time": tf_start,
                        "slot": slot,
                        "x": x,
                        "y": y,
                        "tf_start": tf_start,
                        "tf_end": tf_end
                    })

        return deaths


    # ---------------------------------------------------------
    # Função necessária para extrair mortes por TF
    # ---------------------------------------------------------

    def extract_deaths_from_teamfight(tf):
        deaths = []

        tf_start = tf["start"]
        tf_end = tf["end"]

        for slot, tfp in enumerate(tf["players"]):
            deaths_pos = tfp.get("deaths_pos", {})
            if not deaths_pos:
                continue

            for x_str, ys in deaths_pos.items():
                x = int(x_str)
                for y_str, count in ys.items():
                    y = int(y_str)

                    deaths.append({
                        "time": tf_start,
                        "slot": slot,
                        "x": x,
                        "y": y,
                        "tf_start": tf_start,
                        "tf_end": tf_end
                    })

        return deaths


    # ---------------------------------------------------------
    # TAB 3 — OBSERVER & SENTRY (550x550)
    # ---------------------------------------------------------

    with tab_wards:
        st.header("Observer & Sentry Wards")

        teams = sorted(set(list(data["radiant"].keys()) + list(data["dire"].keys())))
        team = st.selectbox("Selecione o time:", teams, key="ward_team")

        side = st.radio("Selecione o lado:", ["Radiant", "Dire"], key="ward_side")

        if side == "Radiant":
            matches = data["radiant"][team]["matches"]
        else:
            matches = data["dire"][team]["matches"]

        st.write(f"Partidas encontradas: **{len(matches)}**")

        # RANGE REAL DE TEMPOS DAS WARDS
        all_times = []
        all_wards = []

        for fname in matches:
            data_match = load_match(fname)
            wards = extract_wards_from_match(data_match, side)
            all_wards.extend(wards)
            for w in wards:
                all_times.append(w["time"])

        if all_times:
            real_min = min(all_times)
            real_max = max(all_times)
        else:
            real_min, real_max = -90, 60

        real_min_min = real_min / 60
        real_max_min = real_max / 60

        time_min, time_max = st.slider(
            "Intervalo de tempo (minutos)",
            min_value=real_min_min,
            max_value=real_max_min,
            value=(real_min_min, real_min_min + 10),
            step=0.1,
            key="ward_time_slider"
        )

        tmin = time_min * 60
        tmax = time_max * 60

        show_deaths_in = st.checkbox("Mostrar mortes DENTRO da visão das wards")
        show_deaths_out = st.checkbox("Mostrar mortes FORA da visão das wards")

        if st.button("Gerar Wards", key="ward_button"):

            deaths_inside = []
            deaths_outside = []

            # ---------------------------------------------------------
            # PROCESSAR TODAS AS MORTES
            # ---------------------------------------------------------
            for fname in matches:
                match = load_match(fname)
                wards_match = extract_wards_from_match(match, side)

                # bounds iguais ao lane_pos
                xs, ys = [], []
                for p in match["players"]:
                    lp = p.get("lane_pos", {})
                    for x_str, ys_dict in lp.items():
                        xs.append(int(x_str))
                        for y_str in ys_dict.keys():
                            ys.append(int(y_str))

                if not xs or not ys:
                    continue

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                for tf in match.get("teamfights", []):
                    tf_start = tf["start"]
                    tf_end = tf["end"]

                    # TF precisa estar dentro do intervalo
                    if tf_end < tmin or tf_start > tmax:
                        continue

                    deaths = extract_deaths_from_teamfight(tf)

                    for d in deaths:
                        dx, dy = d["x"], d["y"]

                        px, py = death_to_pixel_550(dx, dy, x_min, x_max, y_min, y_max)

                        color = "green" if d["slot"] <= 4 else "red"

                        death_seen = False

                        for w in wards_match:
                            w_start = w["time"]
                            if w["left_time"] is not None:
                                w_end = w["left_time"]
                            else:
                                max_duration = 360 if w["type"] == "obs" else 420
                                w_end = w_start + max_duration

                            # ward viva durante TF
                            if tf_end < w_start or tf_start > w_end:
                                continue

                            # ward viva dentro do intervalo do slider
                            alive = not (w_end < tmin or w_start > tmax)
                            if not alive:
                                continue

                            px_w, py_w = raw_to_od(w["x"], w["y"])
                            radius = 69 if w["type"] == "obs" else 43

                            dxp = px - px_w
                            dyp = py - py_w
                            dist = math.sqrt(dxp*dxp + dyp*dyp)

                            if dist <= radius:
                                death_seen = True
                                break

                        death_info = {
                            "px": px,
                            "py": py,
                            "color": color,
                            "slot": d["slot"],
                            "time": d["time"],
                            "x": dx,
                            "y": dy,
                            "match_id": match["match_id"]
                        }

                        if death_seen:
                            deaths_inside.append(death_info)
                        else:
                            deaths_outside.append(death_info)

            # ---------------------------------------------------------
            # ESCOLHER O QUE EXIBIR
            # ---------------------------------------------------------
            if show_deaths_in:
                deaths_to_plot = deaths_inside
                deaths_to_list = deaths_inside
            elif show_deaths_out:
                deaths_to_plot = deaths_outside
                deaths_to_list = deaths_outside
            else:
                deaths_to_plot = []
                deaths_to_list = []

            # ---------------------------------------------------------
            # PLOT FINAL (inalterado)
            # ---------------------------------------------------------
            fig = plot_wards_and_deaths(all_wards, deaths_to_plot, tmin, tmax)
            st.pyplot(fig)

            # ---------------------------------------------------------
            # LISTA DE MORTES ABAIXO DA IMAGEM
            # ---------------------------------------------------------
            if deaths_to_list:
                st.subheader("Lista de mortes:")

                for d in deaths_to_list:
                    player_color = "green" if d["slot"] <= 4 else "red"
                    player_name = match["players"][d["slot"]]["name"]

                    st.markdown(
                        f"<span style='color:{player_color}; font-weight:bold;'>"
                        f"{player_name}</span> — "
                        f"morreu aos **{d['time']}s**, "
                        f"posição (**{d['x']}**, **{d['y']}**), "
                        f"partida **{d['match_id']}**",
                        unsafe_allow_html=True
                    )

    # ---------------------------------------------------------
    # TAB 4 — SMOKE MAP (550x550)
    # ---------------------------------------------------------

    with tab_smoke:

        # caminhos específicos para o arquivo agregado de smokes e minimapa


        # função que gera a figura a partir do aggregated (mesma lógica que você já aprovou)
        def plot_smokes_from_aggregated_for_tab(aggregated_path: str, mapImage_path: str, img_w: int = 550, img_h: int = 550):
            if not os.path.exists(aggregated_path):
                raise FileNotFoundError(f"Arquivo não encontrado: {aggregated_path}")
            if not os.path.exists(mapImage_path):
                raise FileNotFoundError(f"Imagem do mapa não encontrada: {mapImage_path}")

            with open(aggregated_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            game_start_tick = data.get("game_start_tick")
            events = data.get("smoke_events_aggregated", [])

            # bounds do replay (mesmos valores usados no seu script)
            min_x = 8529.375
            max_x = 24554.5625
            min_y = 8292.75
            max_y = 24464.375

            span_x = max_x - min_x
            span_y = max_y - min_y

            radius_world = 1025
            radius_px = radius_world * (img_w / span_x)

            bg = Image.open(mapImage_path).resize((img_w, img_h))
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(bg)
            ax.axis("off")

            TEAM_COLORS = {"Radiant": "green", "Dire": "red", None: "gray", "Unknown": "gray"}
            ACTIVATION_COLOR = "#800080"  # roxo
            DELTA_COLOR = "#FFD700"       # amarelo
            FONT_SIZE = 5  # metade do tamanho anterior, conforme pedido

            def _tick_to_seconds(tick, game_start_tick):
                try:
                    return (int(tick) - int(game_start_tick)) / 30.0
                except Exception:
                    return None

            def _fmt_time(ts):
                if ts is None:
                    return "-"
                try:
                    return f"{round(float(ts), 1)}s"
                except Exception:
                    return "-"

            def _fmt_delta(d):
                if d is None:
                    return "-"
                try:
                    return f"{round(float(d), 1)}s"
                except Exception:
                    return "-"

            for rec in events:
                pos = rec.get("activator_pos")
                if not pos or not isinstance(pos, (list, tuple)) or len(pos) < 2:
                    continue
                try:
                    x_world = float(pos[0])
                    y_world = float(pos[1])
                except Exception:
                    continue

                source = rec.get("source", {}) if isinstance(rec.get("source"), dict) else {}
                tick = source.get("tick") or rec.get("tick")
                t_seconds = None
                if source.get("time_seconds") is not None:
                    t_seconds = source.get("time_seconds")
                elif tick is not None and game_start_tick is not None:
                    t_seconds = _tick_to_seconds(tick, game_start_tick)

                dispelled = rec.get("dispelled", []) if isinstance(rec.get("dispelled"), list) else []
                disp_times = []
                for d in dispelled:
                    if not isinstance(d, dict):
                        continue
                    ts = d.get("time_seconds")
                    if ts is None and d.get("tick") is not None and game_start_tick is not None:
                        ts = _tick_to_seconds(d.get("tick"), game_start_tick)
                    if ts is not None:
                        try:
                            disp_times.append(float(ts))
                        except Exception:
                            pass
                disp_times.sort()

                if len(disp_times) >= 2:
                    delta = disp_times[-1] - disp_times[0]
                elif len(disp_times) == 1:
                    delta = 0.0
                else:
                    delta = None

                team = rec.get("team")
                color = TEAM_COLORS.get(team, TEAM_COLORS[None])

                nx = (x_world - min_x) / span_x
                ny = (y_world - min_y) / span_y
                px = nx * img_w
                py = (1 - ny) * img_h

                ax.scatter(px, py, c=color, s=60, edgecolors="black", linewidths=0.6)

                a_txt = _fmt_time(t_seconds)
                d_txt = _fmt_delta(delta)

                # activation (roxo) — acima do ponto
                ax.text(px + 6, py - 2, a_txt, color=ACTIVATION_COLOR, fontsize=FONT_SIZE, weight="bold",
                        bbox=dict(alpha=0.6, edgecolor="none", pad=0.2))

                # delta (amarelo) — abaixo do ponto
                ax.text(px + 6, py + 6, d_txt, color=DELTA_COLOR, fontsize=FONT_SIZE, weight="bold",
                        bbox=dict(alpha=0.6, edgecolor="none", pad=0.2))

                circle = Circle((px, py), radius_px, edgecolor=color, facecolor="none", alpha=0.7, linewidth=1)
                ax.add_patch(circle)

            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)
            ax.set_aspect('equal', adjustable='box')
            ax.axis("off")
            return fig

        st.header("Smoke map — 8785561247")
        st.write("Plota smokes a partir de `smoke_events_aggregated.json`. Cores por team; texto: ativação (roxo) e delta (amarelo).")

        if not os.path.exists(SMOKE_AGG_PATH):
            st.error(f"Arquivo de smokes não encontrado: {SMOKE_AGG_PATH}")
        elif not os.path.exists(SMOKE_MAPImage):
            st.error(f"Imagem do minimapa não encontrada: {SMOKE_MAPImage}")
        else:
            try:
                fig = plot_smokes_from_aggregated_for_tab(SMOKE_AGG_PATH, SMOKE_MAPImage, img_w=IMG_WARDS, img_h=IMG_WARDS)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.exception(e)
