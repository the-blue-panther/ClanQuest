import numpy as np
import pygame
import sys
import math

from stable_baselines3 import PPO
from src.environment.gym_wrapper import MultiAgentSimulatedWorldEnv
from src.agents.emotion import Emotion
from src.environment.world import Season

# =========================================
# CONFIG
# =========================================
GRID_SIZE = 50
CELL_SIZE = 15  # Smaller cells for 50x50
PANEL_WIDTH = 350
FPS = 12

WIDTH = GRID_SIZE * CELL_SIZE + PANEL_WIDTH
HEIGHT = GRID_SIZE * CELL_SIZE

# Colors
BG_COLOR = (10, 10, 15)
FOG_COLOR = (5, 5, 10, 200) 
OBSTACLE_COLOR = (60, 60, 80)
TEXT_COLOR = (255, 255, 255)
ACCENT_COLOR = (0, 255, 255)

CLAN_COLORS = {
    0: (80, 160, 220),   # Soothing Blue
    1: (160, 160, 160), # Soothing Gray
    2: (220, 160, 80),   # Soothing Peach/Orange
}

EMOTION_COLORS = {
    Emotion.CALM: (100, 150, 200),
    Emotion.CONFIDENT: (50, 220, 100), # Bright Green Halo
    Emotion.STRESSED: (255, 180, 50),  # Orange Halo
    Emotion.FEARFUL: (255, 80, 80),    # Red Halo
}

SEASON_COLORS = {
    Season.SPRING: (100, 255, 100, 20),
    Season.SUMMER: (255, 255, 100, 20),
    Season.AUTUMN: (255, 150, 50, 20),
    Season.WINTER: (150, 150, 255, 20),
}

# =========================================
# INIT
# =========================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Emergent MARL: Geopolitical Conflict & Cooperation")
font = pygame.font.SysFont("consolas", 12)
small_font = pygame.font.SysFont("consolas", 11)
bold_font = pygame.font.SysFont("consolas", 14, bold=True)
clock = pygame.time.Clock()

# =========================================
# ENV
# =========================================
env = MultiAgentSimulatedWorldEnv(grid_size=GRID_SIZE, seed=42)
obs_dict, _ = env.reset()

MODEL_PATH = "models/checkpoints_final/ppo_final_4800000_steps.zip"
use_model = False
is_deterministic = False

try:
    model = PPO.load(MODEL_PATH)
    print(f"Loaded trained model: {MODEL_PATH}")
    use_model = True # Default to True if loaded
except Exception as e:
    print(f"Failed to load model ({e}), using random actions.")
    model = None
    use_model = False

# =========================================
# DRAW HELPERS
# =========================================
def draw_world(world):
    # Territory layer
    for (x, y), cid in world.territory_map.items():
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = CLAN_COLORS[cid]
        s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        s.fill((*color, 80)) # Increased alpha from 45 to 80
        screen.blit(s, rect)
        # Subtle border to distinguish cells
        pygame.draw.rect(screen, (*color, 120), rect, 1)

    # Obstacles and Resources
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if world.is_obstacle((x, y)):
                pygame.draw.rect(screen, (20, 20, 20), rect)
            elif (x, y) in world.resource_fields:
                field = world.resource_fields[(x, y)]
                if field.intensity > 0:
                    val = field.intensity / field.max_intensity
                    alpha = int(100 + val * 155) # Dimmer when low
                    color = (0, 255, 200)
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    pygame.draw.circle(s, (*color, alpha), (CELL_SIZE//2, CELL_SIZE//2), CELL_SIZE // 2 - 1)
                    screen.blit(s, rect)

    # Seasonal Tint
    season_surf = pygame.Surface((GRID_SIZE * CELL_SIZE, HEIGHT), pygame.SRCALPHA)
    season_surf.fill(SEASON_COLORS[world.current_season])
    screen.blit(season_surf, (0, 0))

def draw_agents(agents):
    for aid, agent in agents.items():
        if not agent.get("is_alive", True):
            continue
            
        x, y = agent["position"]
        cx = x * CELL_SIZE + CELL_SIZE // 2
        cy = y * CELL_SIZE + CELL_SIZE // 2
        
        clan_color = CLAN_COLORS[agent["clan_id"]]
        emo_color = EMOTION_COLORS[agent["emotion"]]
        
        # Halo (Emotion Visual)
        if agent["emotion"] == Emotion.CONFIDENT:
            # Steady green halo
            pygame.draw.circle(screen, (*EMOTION_COLORS[Emotion.CONFIDENT], 100), (cx, cy), CELL_SIZE, 2)
        elif agent["emotion"] == Emotion.STRESSED:
            # Steady orange halo
            pygame.draw.circle(screen, (*EMOTION_COLORS[Emotion.STRESSED], 100), (cx, cy), CELL_SIZE, 1)
        elif agent["emotion"] == Emotion.FEARFUL:
            # Jittery red halo
            ox, oy = np.random.randint(-2, 3, 2)
            pygame.draw.circle(screen, (*EMOTION_COLORS[Emotion.FEARFUL], 120), (cx + ox, cy + oy), CELL_SIZE, 1)

        # Agent body
        pygame.draw.circle(screen, clan_color, (cx, cy), CELL_SIZE // 2)
        
        # Leader Marker (White Core)
        if agent["is_leader"]:
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy), CELL_SIZE // 4)

def draw_dashboard(env):
    x0 = GRID_SIZE * CELL_SIZE + 15
    y = 15
    
    def line(txt, color=TEXT_COLOR, bold=False, indent=0):
        nonlocal y
        f = bold_font if bold else font
        surf = f.render(txt, True, color)
        screen.blit(surf, (x0 + indent, y))
        y += 18

    line("GEOPOLITICAL ENGINE v2.0", color=ACCENT_COLOR, bold=True)
    line(f"Step: {env.current_step}/{env.max_steps} | {env.world.current_season.name}")
    
    # Model Status
    status_str = "TRAINED MODEL" if use_model else "RANDOM WALKS"
    status_color = (100, 255, 100) if use_model else (255, 100, 100)
    det_str = "(DET)" if is_deterministic and use_model else "(STOCH)" if use_model else ""
    line(f"MODE: {status_str} {det_str}", color=status_color, bold=True)
    line("-" * 38)
    
    # CLAN STATS
    for cid, clan in env.world.clans.items():
        color = CLAN_COLORS[cid]
        # Clan Analytics
        living_agents = [aid for aid in clan.agents if env.agents[aid].get("is_alive", True)]
        pop_count = len(living_agents)
        
        line(f"CLAN {cid} ({clan.clan_type.name})", color=color, bold=True)
        line(f" Population: {pop_count}/{len(clan.agents)} | Comfort: {clan.comfort_level:.1f}", indent=10)
        line(f" Territory: {len(clan.territory)} cells", indent=10)
        
        # Leader & Analytics
        leader_id = next((aid for aid in clan.agents if env.agents[aid]["is_leader"]), None)
        stats = env.get_clan_analytics(cid)
        
        if leader_id and env.agents[leader_id].get("is_alive", True):
            emo = env.agents[leader_id]["emotion"]
            line(f" Leader: {emo.name} (Alive)", indent=10)
        elif leader_id:
            line(f" Leader: DECEASED", color=(255, 50, 50), indent=10)
            
        if stats:
            def risk_nature(w):
                if w < 0.7: return "Taking"
                if w > 1.3: return "Averse"
                return "Neutral"
            line(f" Clan Avg: {stats['clan_emotion'].name} | {risk_nature(stats['avg_risk_weight'])} Risk", indent=10, color=(180, 180, 220))
        
        y += 10
    
    line("-" * 38)
    
    # LEGEND
    line("LEGEND & EXPLANATIONS", color=ACCENT_COLOR, bold=True)
    s_line = lambda t, c: line(t, color=c, indent=5)
    s_line("● Cyan/Orange/Grey: Clans", TEXT_COLOR)
    s_line("○ White Core: Clan Leader", (255, 255, 255))
    s_line("◎ Green Halo: Confident", (50, 220, 100))
    s_line("◎ Orange Halo: Stressed", (255, 180, 50))
    s_line("◎ Red Jitter: Fearful", (255, 80, 80))
    s_line("◆ Resource: Intensity = Opacity", (0, 255, 200))
    
    line("CONTROLS:", color=ACCENT_COLOR, bold=True)
    s_line("[M] Toggle Model/Random", TEXT_COLOR)
    s_line("[D] Toggle Deterministic", TEXT_COLOR)
    
    line("-" * 38)
    
    # EVENT LOG
    line("RECENT GLOBAL EVENTS", color=ACCENT_COLOR, bold=True)
    for msg in env.event_log[-5:]:
        color = (255, 255, 100) if "WAR" in msg else (100, 255, 100) if "DISCOVERY" in msg else TEXT_COLOR
        line(f"> {msg}", color=color, indent=5)

# =========================================
# MAIN LOOP
# =========================================
running = True
while running:
    clock.tick(FPS)
    screen.fill(BG_COLOR)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m and model:
                use_model = not use_model
            if event.key == pygame.K_d:
                is_deterministic = not is_deterministic

    # Get actions
    actions = {}
    if model and use_model:
        for aid, obs in obs_dict.items():
            action, _ = model.predict(obs, deterministic=is_deterministic)
            actions[aid] = int(action)
    else:
        actions = {aid: env.action_space.sample() for aid in env.agents}
    
    obs_dict, rewards, term, trunc, _ = env.step(actions)

    draw_world(env.world)
    draw_agents(env.agents)
    draw_dashboard(env)

    # Only reset if the Era is truncated (2000 steps)
    if any(trunc.values()):
        # Show Reset Overlay
        reset_surf = pygame.Surface((GRID_SIZE * CELL_SIZE, HEIGHT), pygame.SRCALPHA)
        reset_surf.fill((0, 0, 0, 180))
        screen.blit(reset_surf, (0, 0))
        
        msg = bold_font.render("SIMULATION ERA COMPLETE - RESETTING WORLD", True, (255, 255, 100))
        msg_rect = msg.get_rect(center=( (GRID_SIZE * CELL_SIZE)//2, HEIGHT//2))
        screen.blit(msg, msg_rect)
        pygame.display.flip()
        pygame.time.wait(2000) # Wait 2 seconds so user sees it
        
        obs_dict, _ = env.reset()

    pygame.display.flip()

pygame.quit()
sys.exit()
