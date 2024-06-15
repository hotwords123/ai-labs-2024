"""
SGF Format: https://homepages.cwi.nl/~aeb/go/misc/sgf.html
"""

import dataclasses
import re


class SGFObject:
    ESCAPE_CHARS = re.compile(r"[\]:\\]")

    def to_sgf(self) -> str:
        raise NotImplementedError

    @classmethod
    def escape_text(cls, text: str) -> str:
        return cls.ESCAPE_CHARS.sub(lambda c: f"\\{c.group()}", text)


@dataclasses.dataclass
class SGFNode(SGFObject):
    properties: dict[str, list[str]] = dataclasses.field(default_factory=dict)

    def get(self, key: str) -> str | None:
        if key in self.properties:
            return self.properties[key][0]
        return None

    def get_all(self, key: str) -> list[str]:
        return self.properties.get(key, [])

    def set(self, key: str, value: str):
        self.properties[key] = [value]

    def set_all(self, key: str, values: list[str]):
        self.properties[key] = values

    def to_sgf(self) -> str:
        return ";" + "".join(
            key + "".join(f"[{self.escape_text(value)}]" for value in values)
            for key, values in self.properties.items()
        )


@dataclasses.dataclass
class SGFSubtree(SGFObject):
    nodes: list[SGFNode] = dataclasses.field(default_factory=list)
    tails: list["SGFSubtree"] = dataclasses.field(default_factory=list)

    def to_sgf(self) -> str:
        return "".join(node.to_sgf() for node in self.nodes) + "".join(
            f"({subtree.to_sgf()})" for subtree in self.tails
        )


@dataclasses.dataclass
class SGFGame(SGFObject):
    root: SGFNode = dataclasses.field(default_factory=SGFNode)
    tree: SGFSubtree = dataclasses.field(default_factory=SGFSubtree)

    def to_sgf(self) -> str:
        return f"({self.root.to_sgf()}{self.tree.to_sgf()})\n"


@dataclasses.dataclass
class SGFCollection:
    games: list[SGFGame] = dataclasses.field(default_factory=list)

    def to_sgf(self) -> str:
        return "".join(game.to_sgf() for game in self.games)


class SGFParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    @property
    def _done(self) -> bool:
        return self.pos >= len(self.text)

    def parse(self) -> SGFCollection:
        collection = self._parse_collection()
        self._skip_whitespace()
        assert self._done, f"Expected end of input at position {self.pos}"
        return collection

    def _skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self, n: int = 1, skip_whitespace: bool = True) -> str:
        if skip_whitespace:
            self._skip_whitespace()
        return self.text[self.pos : self.pos + n]

    def _consume(self, n: int = 1, skip_whitespace: bool = True) -> str:
        result = self._peek(n, skip_whitespace)
        self.pos += len(result)
        return result

    def _accept(self, token: str) -> bool:
        if self._peek(len(token)) == token:
            self._consume(len(token))
            return True
        return False

    def _expect(self, token: str):
        assert self._accept(token), f"Expected {token} at position {self.pos}"

    def _parse_collection(self) -> SGFCollection:
        games = []
        while self._peek() == "(":
            games.append(self._parse_game())
        return SGFCollection(games)

    def _parse_game(self) -> SGFGame:
        self._expect("(")
        root = self._parse_node()
        tree = self._parse_subtree()
        self._expect(")")
        return SGFGame(root, tree)

    def _parse_subtree(self) -> SGFSubtree:
        nodes = []
        while self._peek() == ";":
            nodes.append(self._parse_node())
        tails = []
        while self._accept("("):
            tails.append(self._parse_subtree())
            self._expect(")")
        return SGFSubtree(nodes, tails)

    def _parse_node(self) -> SGFNode:
        self._expect(";")
        properties = {}
        while self._peek().isalpha():
            ident = self._parse_prop_ident()
            values = [self._parse_prop_value()]
            while self._peek() == "[":
                values.append(self._parse_prop_value())
            properties[ident] = values
        return SGFNode(properties)

    def _parse_prop_ident(self) -> str:
        ident = self._consume()
        while self._peek(skip_whitespace=False).isalpha():
            ident += self._consume(skip_whitespace=False)
        return ident

    def _parse_prop_value(self) -> str:
        self._expect("[")
        value = ""
        while (c := self._consume(skip_whitespace=False)) != "]":
            assert c, "Unexpected end of input"
            value += c if c != "\\" else self._consume(skip_whitespace=False)
        return value


__all__ = [
    "SGFObject",
    "SGFNode",
    "SGFSubtree",
    "SGFGame",
    "SGFCollection",
    "SGFParser",
]
