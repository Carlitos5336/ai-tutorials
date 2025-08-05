from dataclasses import dataclass, field

@dataclass
class Tags:

    highway: str = ''
    oneway: bool = False
    max_speed: int = 0
    name: str = ''

@dataclass
class Way:

    id: str
    tags: Tags = field(default_factory=Tags)
    nodes: list = field(default_factory=list)
    open: bool = True