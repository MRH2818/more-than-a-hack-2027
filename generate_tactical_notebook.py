"""Generate the tactical_data_analysis.ipynb notebook."""
import json
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_notebook():
    cells = []

    def md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header
    md("""# ⚽ Tactical Data Analysis — More than a Hack 2027

This notebook provides an in-depth exploratory data analysis (EDA) of all **tactical data assets** provided for the **More than a Hack 2027** ball-possession challenge (*FC Barcelona vs Athletic Club*, LaLiga EA Sports 2024-25, Matchday 2).

### The Objective
For each of the **199,319 frames** (25 fps, ~133 minutes) of the tactical recording, assign one label:
* `H`: Home team in possession, ball alive
* `A`: Away team in possession, ball alive
* `D`: Dead ball (ball out of play, foul, restart, etc.)
* `X`: Ungraded / abstain (halftime, pre-match, post-match)

### What this notebook explores:
1. **Asset Discovery & Verification**: Tactical vs. Broadcast timelines, file schemas, and alignment.
2. **Tactical Homography**: Pitch calibration, metric projection ($105\\text{m} \\times 68\\text{m}$), and reprojection error stability.
3. **Automatic Tracking Data (ATD)**: Player slots, speed dynamics, and the **~41% missing ball problem**.
4. **Metrica Smart Tagging**: Inferred tactical phases (`BUILD UP`, `DEFENSIVE BLOCK`, `RECOVERY`, etc.), set pieces, and duration distributions.
5. **Pitch Dynamics & Spatial Formations**: Team centroids, pitch compactness, and ball-player proximity.
6. **Public Ground Truth Alignment**: Empirical analysis of the 5-minute public ground truth against tracking signals to understand `spellF1` behavior.""")

    # Section 1
    md("""---
## 1. Environment & Asset Setup

We resolve data paths using `games.py` and inspect whether tactical or broadcast files are available.
The reference timeline is **tactical**: row $i$ of any submission corresponds to frame $i$ of `BAR-ATH_tactical.mp4`.""")

    code("""import os
import sys
import glob
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Ensure repository root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..') if os.path.basename(os.getcwd()) == 'starter' else os.getcwd())
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import games

print(f"Project root: {ROOT_DIR}")
home_name, away_name = "Barcelona (Home)", "Athletic Club (Away)"
game_cfg = games.GAMES[games.GAME]
fps = game_cfg['fps']
n_frames = game_cfg['n_video_frames']
print(f"Game: {games.GAME} ({home_name} vs {away_name})")
print(f"FPS: {fps}, Video frames: {n_frames:,} ({n_frames / (fps * 60):.1f} mins)")""")

    code("""# Inspect configured tactical assets
tactical_paths = games.asset_paths(source="tactical")
print("Tactical assets status:")
for name, path in tactical_paths.items():
    exists = os.path.isfile(path)
    size_mb = os.path.getsize(path) / 1024**2 if exists else 0
    status = f"EXISTS ({size_mb:.1f} MB)" if exists else "NOT FOUND"
    print(f"  {name:<15} {os.path.basename(path):<38} {status}")

# Check if broadcast assets exist or were placed in the directory
broadcast_paths = games.asset_paths(source="broadcast")
print("\\nBroadcast assets status:")
for name, path in broadcast_paths.items():
    exists = os.path.isfile(path)
    size_mb = os.path.getsize(path) / 1024**2 if exists else 0
    status = f"EXISTS ({size_mb:.1f} MB)" if exists else "NOT FOUND"
    print(f"  {name:<15} {os.path.basename(path):<38} {status}")""")

    # Section 2
    md("""---
## 2. Tactical Homography & Pitch Geometry

The homography matrix $H$ is a $3 \\times 3$ projective transformation mapping normalized pitch coordinates $(x, y) \\in [0, 1] \\times [0, 1]$ directly to video frame pixel coordinates $(u, v)$:

$$\\begin{bmatrix} u' \\\\ v' \\\\ w' \\end{bmatrix} = H \\begin{bmatrix} x \\\\ y \\\\ 1 \\end{bmatrix}, \\quad u = \\frac{u'}{w'}, \\quad v = \\frac{v'}{w'}$$

* **Reprojection Error**: Quantifies calibration fit quality. Lower is better (median $\\approx 0.24$ in live play).
* **Validity Flag**: In the tactical recording, `valid == 1` virtually everywhere because the tactical camera maintains a fixed wide-angle view of the pitch.""")

    code("""# Load tactical homography (or fallback to broadcast homography if testing before tactical download)
homography_path = tactical_paths['homography']
if not os.path.isfile(homography_path):
    # Check data/tactical/ for broadcast homography if present
    alt_homography = os.path.join(ROOT_DIR, "data", "tactical", "BAR-ATH_broadcast_homography.csv")
    if os.path.isfile(alt_homography):
        homography_path = alt_homography
        print(f"Note: Using {os.path.basename(alt_homography)} as homography reference.")
    elif os.path.isfile(broadcast_paths['homography']):
        homography_path = broadcast_paths['homography']
        print(f"Note: Using broadcast homography.")

if os.path.isfile(homography_path):
    df_homography = pd.read_csv(homography_path)
    print(f"Loaded {len(df_homography):,} rows from {os.path.basename(homography_path)}")
    print(df_homography.head(3))
else:
    df_homography = None
    print("Homography file not found. Place bundle files in data/tactical/")""")

    code("""if df_homography is not None:
    valid_count = (df_homography['valid'] == 1).sum()
    total_count = len(df_homography)
    print(f"Valid frames: {valid_count:,} / {total_count:,} ({100 * valid_count / total_count:.2f}%)")
    
    # Analyze reprojection error
    valid_reproj = df_homography.loc[df_homography['valid'] == 1, 'reprojection_error']
    print(f"Reprojection error stats (valid frames):")
    print(f"  Mean:   {valid_reproj.mean():.4f}")
    print(f"  Median: {valid_reproj.median():.4f}")
    print(f"  95th %: {valid_reproj.quantile(0.95):.4f}")
    print(f"  Max:    {valid_reproj.max():.4f}")

    # Plot reprojection error over match timeline
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    ax1.plot(df_homography['frame'] / (25 * 60), df_homography['valid'], color='#2ca02c', alpha=0.7)
    ax1.set_ylabel('Valid (1=Yes)')
    ax1.set_title('Camera Pitch Calibration & Homography Quality Over Time')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df_homography['frame'] / (25 * 60), df_homography['reprojection_error'].clip(upper=10), color='#1f77b4', lw=0.8)
    ax2.axhline(valid_reproj.median(), color='red', linestyle='--', label=f'Median: {valid_reproj.median():.2f}')
    ax2.set_xlabel('Minute of Video')
    ax2.set_ylabel('Reprojection Error (clipped at 10)')
    ax2.set_ylim(-0.5, 10.5)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()""")

    # Section 3
    md("""---
## 3. Automatic Tracking Data (ATD) & Missing Ball Phenomenon

The tracking data is exported in FIFA/EPTS format:
* **Metadata XML**: Team rosters, channel IDs, pitch dimensions ($105.3\\text{m} \\times 68.0\\text{m}$).
* **Raw ATD Text**: Line-delimited stream with 1 record per frame.
  Format: `<frame_id>:<player_track_1>;<player_track_2>;...:<ball>`
  Where each group is `x,y,speed` in $[0, 1]$.

### The Key Challenge: The Missing Ball
In real computer vision tracking, **the ball is occluded or not detected on ~41% of frames**!
A naïve "nearest player to ball" model fails on 41% of frames unless we explicitly model dead-ball conditions or track player collective motions.""")

    code("""# Check available tracking files
atd_xml = tactical_paths['atd_meta']
atd_txt = tactical_paths['atd_raw']

if not (os.path.isfile(atd_xml) and os.path.isfile(atd_txt)):
    # Fallback to broadcast files in data/tactical if present
    alt_xml = os.path.join(ROOT_DIR, "data", "tactical", "BAR-ATH_broadcast_atd.xml")
    alt_txt = os.path.join(ROOT_DIR, "data", "tactical", "BAR-ATH_broadcast_atd.txt")
    if os.path.isfile(alt_xml) and os.path.isfile(alt_txt):
        atd_xml, atd_txt = alt_xml, alt_txt
        print(f"Using {os.path.basename(alt_xml)} and {os.path.basename(alt_txt)}")
    elif os.path.isfile(broadcast_paths['atd_meta']) and os.path.isfile(broadcast_paths['atd_raw']):
        atd_xml, atd_txt = broadcast_paths['atd_meta'], broadcast_paths['atd_raw']
        print(f"Using broadcast ATD assets.")

print(f"Tracking XML: {atd_xml} (Exists: {os.path.isfile(atd_xml)})")
print(f"Tracking TXT: {atd_txt} (Exists: {os.path.isfile(atd_txt)})")""")

    code("""# Parse tracking metadata without external dependencies
def parse_atd_metadata(xml_path):
    if not os.path.isfile(xml_path):
        return None
    root = ET.parse(xml_path).getroot()
    teams = []
    for team in root.iter('Team'):
        teams.append({
            'id': team.get('id'),
            'name': team.findtext('Name') or team.get('id', 'Team')
        })
    players = []
    for p in root.iter('Player'):
        players.append({
            'id': p.get('id'),
            'team_id': p.get('teamId')
        })
    return {'teams': teams, 'players': players}

meta_info = parse_atd_metadata(atd_xml)
if meta_info:
    print(f"Tracking metadata from {os.path.basename(atd_xml)}:")
    for i, t in enumerate(meta_info['teams']):
        n_p = sum(1 for p in meta_info['players'] if p['team_id'] == t['id'])
        print(f"  Team {i} [{t['name']}]: {n_p} assigned track slots")""")

    code("""# Stream sample of raw tracking data to inspect ball detection and speed
def analyze_tracking_sample(txt_path, max_frames=20000):
    if not os.path.isfile(txt_path):
        return None
    
    frame_ids = []
    ball_visible = []
    ball_speeds = []
    player_counts = []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= max_frames:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(':')
            if len(parts) < 3:
                continue
            
            f_id = int(parts[0])
            players_str = parts[1]
            ball_str = parts[2]
            
            # Count tracked players in this frame (non-NaN)
            p_slots = players_str.split(';')
            n_players = sum(1 for p in p_slots if not p.startswith('NaN'))
            
            # Ball status
            ball_vals = ball_str.split(',')
            has_ball = not ball_vals[0].startswith('NaN') and len(ball_vals) >= 2
            b_speed = float(ball_vals[2]) if (has_ball and len(ball_vals) > 2 and ball_vals[2] != 'NaN') else np.nan
            
            frame_ids.append(f_id)
            ball_visible.append(has_ball)
            ball_speeds.append(b_speed)
            player_counts.append(n_players)
            
    return pd.DataFrame({
        'frame_id': frame_ids,
        'ball_visible': ball_visible,
        'ball_speed': ball_speeds,
        'n_players': player_counts
    })

df_sample = analyze_tracking_sample(atd_txt, max_frames=25000)
if df_sample is not None:
    vis_pct = 100 * df_sample['ball_visible'].mean()
    print(f"Analyzed {len(df_sample):,} tracking frames:")
    print(f"  Ball detected: {df_sample['ball_visible'].sum():,} ({vis_pct:.1f}%)")
    print(f"  Ball missing:  {(~df_sample['ball_visible']).sum():,} ({100 - vis_pct:.1f}%)")
    print(f"  Mean active players on pitch: {df_sample['n_players'].mean():.1f}")
    
    # Plot ball detection and player counts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    time_min = df_sample['frame_id'] / (25 * 60)
    
    ax1.plot(time_min, df_sample['ball_visible'].astype(int), color='#d62728', alpha=0.6, lw=0.5)
    ax1.set_ylabel('Ball Visible (1/0)')
    ax1.set_title('Ball Visibility and Active Player Tracking Over Sample Window')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_min, df_sample['n_players'], color='#1f77b4', lw=1.0)
    ax2.axhline(22, color='black', linestyle='--', label='22 players (11v11)')
    ax2.set_xlabel('Minute of Video')
    ax2.set_ylabel('Tracked Players')
    ax2.set_ylim(0, 30)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()""")

    # Section 4
    md("""---
## 4. Metrica Tactical Smart Tagging Analysis

Metrica Sports' **Smart Tagging** automatically annotates tactical phases of play and set pieces directly aligned to the tactical video timeline:
* `<start>` and `<end>` timestamps are in seconds from the first frame of the tactical video.
* Video frame mapping is exact: $\\text{frame} = \\text{round}(\\text{seconds} \\times 25)$.
* Total of **638 phases of play** across the match:
  - `SET PIECES`: Kick-offs, throw-ins, corners, free kicks, goal kicks.
  - `BUILD UP`: Controlled possession starting from the defensive or middle third.
  - `DEFENSIVE BLOCK`: Structured defensive shape without the ball.
  - `RECOVERY`: Regaining the ball / counter-pressing.""")

    code("""# Load and parse Smart Tagging XML
smart_tagging_path = tactical_paths['smart_tagging']

def parse_smart_tagging(xml_path):
    if not os.path.isfile(xml_path):
        return None
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    records = []
    # SportsCode XML structure: <ALL_INSTANCES><instance>...
    for inst in root.iter('instance'):
        code_elem = inst.find('code')
        start_elem = inst.find('start')
        end_elem = inst.find('end')
        
        if code_elem is None or start_elem is None or end_elem is None:
            continue
            
        code = code_elem.text
        start_s = float(start_elem.text)
        end_s = float(end_elem.text)
        
        # Tags (Team, Half, Type, etc.)
        labels = {}
        for label in inst.iter('label'):
            group_elem = label.find('group')
            text_elem = label.find('text')
            if group_elem is not None and text_elem is not None:
                labels[group_elem.text] = text_elem.text
                
        records.append({
            'code': code,
            'start_s': start_s,
            'end_s': end_s,
            'start_frame': int(round(start_s * 25)),
            'end_frame': int(round(end_s * 25)),
            'duration_s': end_s - start_s,
            'team': labels.get('Team', 'None'),
            'half': labels.get('Half', 'None'),
            'type': labels.get('Type', 'None')
        })
        
    return pd.DataFrame(records)

df_tags = parse_smart_tagging(smart_tagging_path)
if df_tags is not None:
    print(f"Loaded {len(df_tags)} tactical phases from Smart Tagging:")
    print(df_tags.head())
else:
    print(f"Smart tagging file not found at {smart_tagging_path}. (Included when full bundle is unpacked)")""")

    code("""if df_tags is not None:
    # Distribution of tactical codes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    code_counts = df_tags['code'].value_counts()
    code_counts.plot(kind='barh', ax=ax1, color='#1f77b4', edgecolor='black')
    ax1.set_title('Frequency of Tactical Phases (Codes)')
    ax1.set_xlabel('Count')
    ax1.grid(axis='x', alpha=0.3)
    
    # Team breakdown for top phases
    top_codes = code_counts.index[:5]
    team_breakdown = df_tags[df_tags['code'].isin(top_codes)].groupby(['code', 'team']).size().unstack(fill_value=0)
    team_breakdown.plot(kind='bar', ax=ax2, edgecolor='black', colormap='Set1')
    ax2.set_title('Top Tactical Phases by Team')
    ax2.set_ylabel('Count')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Duration analysis
    print("\\nTactical Phase Durations (seconds):")
    duration_stats = df_tags.groupby('code')['duration_s'].agg(['count', 'mean', 'median', 'std', 'max'])
    print(duration_stats.sort_values(by='count', ascending=False))""")

    # Section 5
    md("""---
## 5. Spatial Pitch Metrics & Tactical Formations

Using pitch geometry ($105.0\\text{m} \\times 68.0\\text{m}$), we visualize player coordinates, pitch zones, and calculate:
* **Team Centroid**: Center of mass of the 10 outfield players $\\left(\\bar{x} = \\frac{1}{N}\\sum x_i, \\bar{y} = \\frac{1}{N}\\sum y_i\\right)$.
* **Team Spread (Length & Width)**: Standard deviation / bounding box of player coordinates.
* **Proximity to Ball**: Euclidean distance from each player to the ball position.""")

    code("""def draw_pitch(ax=None, length=105.0, width=68.0):
    \"\"\"Draw a regulation football pitch in metres.\"\"\"
    from matplotlib.patches import Circle, Rectangle
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 7))
        
    grass = '#3f8f4f'
    line = '#ffffff'
    
    # Pitch boundary
    ax.add_patch(Rectangle((0, 0), length, width, facecolor=grass, edgecolor=line, lw=1.5, zorder=0))
    # Halfway line & center circle
    ax.plot([length / 2, length / 2], [0, width], color=line, lw=1.5, zorder=1)
    ax.add_patch(Circle((length / 2, width / 2), 9.15, fill=False, edgecolor=line, lw=1.5, zorder=1))
    ax.scatter(length / 2, width / 2, color=line, s=20, zorder=1)
    
    # Penalty and 6-yard boxes
    for side in (0, 1):
        for depth, box_w in ((16.5, 40.32), (5.5, 18.32)):
            x = 0 if side == 0 else length - depth
            y = (width - box_w) / 2
            ax.add_patch(Rectangle((x, y), depth, box_w, fill=False, edgecolor=line, lw=1.5, zorder=1))
            
    ax.set_xlim(-4, length + 4)
    ax.set_ylim(-4, width + 4)
    ax.set_aspect('equal')
    ax.set_xlabel('Pitch Length (m)')
    ax.set_ylabel('Pitch Width (m)')
    return ax

fig, ax = plt.subplots(figsize=(10, 6.5))
draw_pitch(ax)
ax.set_title('Tactical Pitch Coordinates (105m x 68m)')
plt.show()""")

    md("""### Ball Motion Trajectory: First 10 Minutes of the Match

The first-half kickoff takes place at video frame **11,185** (`07:27` from video start).
The first 10 minutes of active play span **15,000 frames** (frames 11,185 to 26,184).

The cell below streams the ball tracking coordinates, projects them to the metric pitch ($105.0\\text{m} \\times 68.0\\text{m}$), and renders the continuous ball motion trajectory on the pitch graphic, color-coded by match minutes from kickoff (0 to 10 min).""")

    code("""from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# Define 10-minute game window from kickoff
kickoff_frame = 11185
duration_mins = 10
n_frames_10m = duration_mins * 60 * 25  # 15,000 frames
end_frame = kickoff_frame + n_frames_10m

ball_records = []
if os.path.isfile(atd_txt):
    with open(atd_txt, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= end_frame:
                break
            if idx >= kickoff_frame:
                parts = line.strip().split(':')
                if len(parts) >= 3:
                    f_id = int(parts[0])
                    b = parts[2].split(',')
                    if len(b) >= 2 and not b[0].startswith('NaN'):
                        bx = float(b[0]) * 105.0  # length in metres
                        by = float(b[1]) * 68.0   # width in metres
                        bspeed = float(b[2]) if (len(b) > 2 and b[2] != 'NaN') else 0.0
                        game_sec = (idx - kickoff_frame) / 25.0
                        ball_records.append((f_id, bx, by, bspeed, game_sec))

df_ball_10m = pd.DataFrame(ball_records, columns=['frame_id', 'x_m', 'y_m', 'speed', 'game_sec'])
print(f"Loaded {len(df_ball_10m):,} ball observations across the first 10 minutes of play ({100 * len(df_ball_10m) / n_frames_10m:.1f}% tracking coverage).")

if not df_ball_10m.empty:
    fig, ax = plt.subplots(figsize=(15, 9))
    draw_pitch(ax, length=105.0, width=68.0)
    
    points = df_ball_10m[['x_m', 'y_m']].values
    times_min = df_ball_10m['game_sec'].values / 60.0
    frame_ids = df_ball_10m['frame_id'].values

    # Build continuous trajectory segments, splitting lines across tracking occlusions (> 5 frames) or teleporting jumps (> 25 m)
    segments = []
    seg_times = []
    for i in range(len(points) - 1):
        if (frame_ids[i+1] - frame_ids[i] <= 5) and (np.hypot(points[i+1, 0] - points[i, 0], points[i+1, 1] - points[i, 1]) < 25):
            segments.append([points[i], points[i+1]])
            seg_times.append(times_min[i])

    norm = Normalize(vmin=0, vmax=10)
    lc = LineCollection(segments, cmap='plasma', norm=norm, linewidths=1.4, alpha=0.85, zorder=3)
    lc.set_array(np.array(seg_times))
    line = ax.add_collection(lc)

    # Plot start (kickoff) and end (10-min mark)
    ax.scatter([df_ball_10m['x_m'].iloc[0]], [df_ball_10m['y_m'].iloc[0]], color='cyan', s=160, edgecolors='black', lw=1.8, zorder=5, label='Kickoff (Minute 0:00)')
    ax.scatter([df_ball_10m['x_m'].iloc[-1]], [df_ball_10m['y_m'].iloc[-1]], color='yellow', s=160, edgecolors='black', lw=1.8, zorder=5, label='10-Minute Mark')

    # Add match time colorbar
    cbar = plt.colorbar(line, ax=ax, orientation='horizontal', pad=0.06, fraction=0.046)
    cbar.set_label('Elapsed Match Time (Minutes from Kickoff: 0 to 10)', fontsize=11, fontweight='bold')
    cbar.set_ticks(np.arange(0, 11, 1))

    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.set_title('Ball Motion Trajectory Throughout the First 10 Minutes of the Match\\n(FC Barcelona vs Athletic Club — First Half Kickoff to Minute 10)', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()""")

    # Section 6
    md("""---
## 6. Ground Truth Possession Alignment & Error Analysis

The challenge repository provides a **public 5-minute ground truth** covering frames **11,185 to 18,684** (7,500 frames) from the first half:
* Ground truth contains **17 distinct possessions** with a mean duration of $14.19$ seconds.
* The evaluation ranking metric is **`spellF1`**, which evaluates whether predicted possession spells match true spells with an IoU threshold $\\ge 0.5$, penalizing flickering and spurious switches.""")

    code("""gt_path = os.path.join(ROOT_DIR, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
meta_path = os.path.join(ROOT_DIR, "evaluation", "gt", "BAR-ATH_possession_gt_public_meta.json")

if os.path.isfile(gt_path) and os.path.isfile(meta_path):
    df_gt = pd.read_csv(gt_path)
    with open(meta_path, 'r') as f:
        gt_meta = json.load(f)
        
    start_frame = gt_meta['public_window']['first_video_frame']
    end_frame = gt_meta['public_window']['last_video_frame']
    
    df_window = df_gt[(df_gt['frame'] >= start_frame) & (df_gt['frame'] <= end_frame)].copy()
    print(f"Ground Truth Public Window:")
    print(f"  Frames: {start_frame:,} to {end_frame:,} ({len(df_window):,} frames)")
    print(f"  Possessions: {gt_meta['n_possessions']}, Mean Duration: {gt_meta['mean_possession_duration_s']:.1f} s")
    print("\\nLabel distribution in public window:")
    print(df_window['label'].value_counts())
else:
    df_window = None
    print("Public ground truth files not found.")""")

    code("""if df_window is not None:
    # Visualize ground truth label timeline
    fig, ax = plt.subplots(figsize=(14, 3))
    
    color_map = {'H': '#d62728', 'A': '#1f77b4', 'D': '#7f7f7f', 'X': '#bcbd22'}
    colors = [color_map[l] for l in df_window['label']]
    
    time_s = (df_window['frame'] - start_frame) / 25
    ax.scatter(time_s, [1] * len(time_s), c=colors, marker='|', s=200, alpha=0.9)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#d62728', lw=4, label='H (Home Possession)'),
        Line2D([0], [0], color='#1f77b4', lw=4, label='A (Away Possession)'),
        Line2D([0], [0], color='#7f7f7f', lw=4, label='D (Dead Ball)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_yticks([])
    ax.set_xlabel('Elapsed Time in Public Window (seconds)')
    ax.set_title('Ground Truth Possession Spells (First 5 Minutes of Play)')
    plt.tight_layout()
    plt.show()""")

    # Baseline comparison
    md("""### Comparing Baseline Variants & The spellF1 Metric
The repository provides 4 baseline submissions differing only by their majority-vote smoothing window:
* `baseline_raw` (window = 0): Extremely noisy, creates 131 false possessions.
* `baseline_smooth25` (window = $\\pm 1\\text{s}$): Reduces noise, 36 possessions.
* `baseline_smooth50` (window = $\\pm 2\\text{s}$): The published floor (`spellF1 = 0.2433` on full match).
* `baseline_smooth125` (window = $\\pm 5\\text{s}$): Over-smooths across rapid handovers.""")

    code("""# Inspect baseline reports
reports_dir = os.path.join(ROOT_DIR, "evaluation", "reports")
report_files = sorted(glob.glob(os.path.join(reports_dir, "*.json")))

if report_files:
    rows = []
    for r_path in report_files:
        with open(r_path, 'r') as f:
            data = json.load(f)
            name = os.path.splitext(os.path.basename(r_path))[0]
            rows.append({
                'Submission': name,
                'macroF1': data.get('macro_f1', np.nan),
                'spellF1 (headline)': data.get('spell_f1_at_headline', np.nan),
                'Predicted Possessions': data.get('n_pred_spells', np.nan),
                'Mean Duration (s)': data.get('mean_spell_duration_s', np.nan),
                'Alive/Dead F1': data.get('alive_dead_f1', np.nan)
            })
    df_board = pd.DataFrame(rows).sort_values(by='spellF1 (headline)', ascending=False)
    print("Baseline Performance Comparison on Public Window:")
    print(df_board.to_string(index=False))
    
    # Plot spellF1 vs macroF1 comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df_board))
    w = 0.35
    ax.bar(x - w/2, df_board['macroF1'], width=w, label='macroF1 (frame-level)', color='#aec7e8', edgecolor='black')
    ax.bar(x + w/2, df_board['spellF1 (headline)'], width=w, label='spellF1 (spell-level IoU>=0.5)', color='#1f77b4', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(df_board['Submission'], rotation=20, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('macroF1 vs. spellF1: Why Possession Structure Dictates Ranking')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()""")

    # Section 7
    md("""---
## 7. Strategic Takeaways for Developing Your Solution

From our tactical analysis across homography, tracking, tagging, and ground truth, here are key strategies to maximize `spellF1`:

1. **Dead Ball Detection (`D`) is the Biggest Blindspot**:
   - The simple nearest-player baseline only detects `D` when the ball is lost by tracking.
   - Genuine dead balls can be detected via:
     - Ball velocity $\\approx 0$ near boundaries or restarts.
     - Referee position / whistle proximity.
     - Smart Tagging `SET PIECES` intervals, which mark kickoffs, corners, throw-ins, and free kicks.
2. **Missing Ball Interpolation**:
   - During the 41% of frames where the ball is invisible, avoid switching possession rapidly. Carry forward controlled possession or use team convex hull centroid momentum.
3. **Temporal Modeling Over Frame-by-Frame Argmax**:
   - Instead of independent frame predictions followed by ad-hoc smoothing, use a **Hidden Markov Model (HMM)**, **Semi-Markov model**, or **Temporal Convolutional Network (TCN)** with transition penalties that mirror realistic football possession length ($14\\text{s}$ mean duration).
4. **Validation Hygiene**:
   - Always run `python evaluation/validate_submission.py <your_file>.csv` before submitting to guarantee exact 199,319-row formatting!""")

    nb_data = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    output_root = os.path.join(ROOT_DIR, "tactical_data_analysis.ipynb")
    output_starter = os.path.join(ROOT_DIR, "starter", "04_tactical_data_analysis.ipynb")

    with open(output_root, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)
    print(f"Created {output_root}")

    with open(output_starter, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)
    print(f"Created {output_starter}")

if __name__ == "__main__":
    create_notebook()
