"""Small SYNTHETIC multi-hop fixture for unit tests / smoke runs.

All entities are invented. Results on this fixture must NEVER be reported as benchmark results.
"""
from __future__ import annotations

from .preprocessing import assign_splits, build_passage, dedupe_passages
from .schema import Dataset, Example

# (id, question, answer, qtype, supporting [(title,text)], distractors [(title,text)])
_RAW = [
    ("f1", "Which country was the author of The Silent Orchard born in?", "Eldoria", "bridge",
     [("The Silent Orchard", "The Silent Orchard is a novel written by Mira Valdoren."),
      ("Mira Valdoren", "Mira Valdoren was born in Eldoria, a small coastal nation.")],
     [("Harbor Lights", "Harbor Lights is a novel written by Tomas Brenner.")]),
    ("f2", "In which city is the university where the inventor of the Kestrel Engine studied located?", "Marlowe", "bridge",
     [("Kestrel Engine", "The Kestrel Engine was invented by Lena Ostrow."),
      ("Lena Ostrow", "Lena Ostrow studied at Halden University."),
      ("Halden University", "Halden University is located in the city of Marlowe.")],
     [("Orwick College", "Orwick College is located in the city of Dunmere.")]),
    ("f3", "Who founded the company that manufactures the Zephyr Drone?", "Priya Natarajan", "bridge",
     [("Zephyr Drone", "The Zephyr Drone is manufactured by Aerowing Corporation."),
      ("Aerowing Corporation", "Aerowing Corporation was founded by Priya Natarajan in 1998.")],
     [("Skyline Dynamics", "Skyline Dynamics was founded by Oskar Lindqvist in 2004.")]),
    ("f4", "What is the capital of the country where the Blue Ridge Festival is held?", "Valmora", "bridge",
     [("Blue Ridge Festival", "The Blue Ridge Festival is held annually in Taloria."),
      ("Taloria", "Taloria is a mountainous country whose capital is Valmora.")],
     [("Eldoria", "Eldoria is a small coastal nation whose capital is Port Sennet.")]),
    ("f5", "Which river flows through the city where Dr. Henrik Rask was born?", "Aldane River", "bridge",
     [("Henrik Rask", "Dr. Henrik Rask was born in Quillon."),
      ("Quillon", "Quillon is a city on the Aldane River.")],
     [("Marlowe", "Marlowe is a city on the Torrin River.")]),
    ("f6", "Who is older, Mira Valdoren or Lena Ostrow?", "Lena Ostrow", "comparison",
     [("Mira Valdoren age", "Mira Valdoren was born in 1975."),
      ("Lena Ostrow age", "Lena Ostrow was born in 1968.")],
     [("Tomas Brenner", "Tomas Brenner was born in 1981.")]),
    ("f7", "Are Halden University and Orwick College located in the same city?", "no", "comparison",
     [("Halden University", "Halden University is located in the city of Marlowe."),
      ("Orwick College", "Orwick College is located in the city of Dunmere.")],
     [("Quillon", "Quillon is a city on the Aldane River.")]),
    ("f8", "When was Aerowing Corporation founded?", "1998", "single_hop",
     [("Aerowing Corporation", "Aerowing Corporation was founded by Priya Natarajan in 1998.")],
     [("Skyline Dynamics", "Skyline Dynamics was founded by Oskar Lindqvist in 2004.")]),
    ("f9", "What is the capital of Taloria?", "Valmora", "single_hop",
     [("Taloria", "Taloria is a mountainous country whose capital is Valmora.")],
     [("Eldoria", "Eldoria is a small coastal nation whose capital is Port Sennet.")]),
    ("f10", "Who wrote Harbor Lights?", "Tomas Brenner", "single_hop",
     [("Harbor Lights", "Harbor Lights is a novel written by Tomas Brenner.")],
     [("The Silent Orchard", "The Silent Orchard is a novel written by Mira Valdoren.")]),
    ("f11", "Which company manufactures the device invented by the founder of Skyline Dynamics?", "Northgate Works", "bridge",
     [("Skyline Dynamics", "Skyline Dynamics was founded by Oskar Lindqvist in 2004."),
      ("Oskar Lindqvist", "Oskar Lindqvist invented the Falcon Radar, which is manufactured by Northgate Works.")],
     [("Zephyr Drone", "The Zephyr Drone is manufactured by Aerowing Corporation.")]),
    ("f12", "What is the capital of the country where Mira Valdoren was born?", "Port Sennet", "bridge",
     [("Mira Valdoren", "Mira Valdoren was born in Eldoria, a small coastal nation."),
      ("Eldoria", "Eldoria is a small coastal nation whose capital is Port Sennet.")],
     [("Taloria", "Taloria is a mountainous country whose capital is Valmora.")]),
]


def load_fixture(seed: int = 42, max_samples: int | None = None) -> Dataset:
    raw = _RAW[:max_samples] if max_samples else _RAW
    examples, passages = [], []
    for eid, q, a, qt, sup, dis in raw:
        gold = []
        for title, text in sup:
            p = build_passage(title, text, {"supporting": True})
            passages.append(p)
            gold.append(p.pid)
        for title, text in dis:
            passages.append(build_passage(title, text))
        examples.append(Example(eid=eid, question=q, answer=a, qtype=qt, gold_pids=gold))
    assign_splits(examples, seed)
    # NB: 'supporting' meta flag is stripped so it cannot be read by retrievers
    passages = dedupe_passages(passages)
    for p in passages:
        p.meta = {}
    return Dataset("fixture", "synthetic (src/data/fixture.py)", "v1", examples, passages, synthetic=True)
