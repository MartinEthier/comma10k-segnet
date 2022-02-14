"""
From comma10k repo:
1 - #402020 - road (all parts, anywhere nobody would look at you funny for driving)
2 - #ff0000 - lane markings (don't include non lane markings like turn arrows and crosswalks)
3 - #808060 - undrivable
4 - #00ff66 - movable (vehicles and people/animals)
5 - #cc00ff - my car (and anything inside it, including wires, mounts, etc. No reflections)
"""

# Class colors are in RGB
MOVEABLE_IDX = 3
CMAP = {
    "road": (64, 32, 32),
    "lane_markings": (255, 0, 0),
    "undrivable": (128, 128, 96),
    "movable": (0, 255, 102),
    "my_car": (204, 0, 255)
}
