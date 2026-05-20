"""Game mapping for AtariBench Dopamine baselines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DopamineGame:
    slug: str
    dopamine_name: str


ATARIBENCH_GAMES: tuple[str, ...] = (
    "air_raid",
    "assault",
    "beam_rider",
    "boxing",
    "breakout",
    "demon_attack",
    "fishing_derby",
    "freeway",
    "gopher",
    "ice_hockey",
    "journey_escape",
    "name_this_game",
    "pacman",
    "phoenix",
    "qbert",
    "riverraid",
    "robotank",
    "seaquest",
    "tennis",
    "time_pilot",
)

EXTRA_GAMES: tuple[str, ...] = (
    "laser_gates",
)

DEFAULT_EVAL_GAMES: tuple[str, ...] = ATARIBENCH_GAMES

GAME_TO_DOPAMINE: dict[str, DopamineGame] = {
    "air_raid": DopamineGame("air_raid", "AirRaid"),
    "assault": DopamineGame("assault", "Assault"),
    "beam_rider": DopamineGame("beam_rider", "BeamRider"),
    "boxing": DopamineGame("boxing", "Boxing"),
    "breakout": DopamineGame("breakout", "Breakout"),
    "demon_attack": DopamineGame("demon_attack", "DemonAttack"),
    "fishing_derby": DopamineGame("fishing_derby", "FishingDerby"),
    "freeway": DopamineGame("freeway", "Freeway"),
    "gopher": DopamineGame("gopher", "Gopher"),
    "ice_hockey": DopamineGame("ice_hockey", "IceHockey"),
    "journey_escape": DopamineGame("journey_escape", "JourneyEscape"),
    "laser_gates": DopamineGame("laser_gates", "LaserGates"),
    "name_this_game": DopamineGame("name_this_game", "NameThisGame"),
    "pacman": DopamineGame("pacman", "MsPacman"),
    "phoenix": DopamineGame("phoenix", "Phoenix"),
    "qbert": DopamineGame("qbert", "Qbert"),
    "riverraid": DopamineGame("riverraid", "Riverraid"),
    "robotank": DopamineGame("robotank", "Robotank"),
    "seaquest": DopamineGame("seaquest", "Seaquest"),
    "tennis": DopamineGame("tennis", "Tennis"),
    "time_pilot": DopamineGame("time_pilot", "TimePilot"),
}


def normalize_game(game: str) -> DopamineGame:
    """Accept an AtariBench slug or a Dopamine/ALE game name."""

    key = game.strip()
    slug_key = key.lower().replace("-", "_")
    if slug_key in GAME_TO_DOPAMINE:
        return GAME_TO_DOPAMINE[slug_key]

    compact = key.lower().replace("_", "").replace("-", "")
    for spec in GAME_TO_DOPAMINE.values():
        if compact == spec.dopamine_name.lower():
            return spec
    raise KeyError(f"Unknown AtariBench/Dopamine game: {game}")
