"""Phase 9 - person search by physical description (clothing colors, height).

Turns a person crop into a small human-readable description (dominant
clothing colors + apparent height class) and scores gallery subjects
against a free-text query like "red shirt, tall". Colors come from an HSV
quantization of the torso region, so they survive lighting shifts better
than raw RGB; height is the bbox height normalized by the frame height
(an apparent-height heuristic - the same person closer to the camera reads
taller, which the height classes are tuned to tolerate).
"""
from __future__ import annotations

import re

import cv2
import numpy as np

# (hue range, saturation floor, value floor) -> name. Hue in OpenCV is 0..179.
_HSV_BINS = [
    ((0, 7), 90, 40, "red"),
    ((168, 179), 90, 40, "red"),
    ((8, 24), 80, 40, "orange"),
    ((25, 34), 60, 40, "yellow"),
    ((35, 84), 50, 35, "green"),
    ((85, 99), 60, 35, "cyan"),
    ((100, 124), 60, 35, "blue"),
    ((125, 148), 50, 35, "purple"),
    ((149, 167), 70, 40, "pink"),
]
_GRAY_FLOOR = 40  # below this saturation everything is gray/black/white


def _color_name(h: int, s: int, v: int) -> str:
    if s < _GRAY_FLOOR:
        if v < 60:
            return "black"
        if v > 200:
            return "white"
        return "gray"
    for (lo, hi), smin, vmin, name in _HSV_BINS:
        if lo <= h <= hi and s >= smin and v >= vmin:
            return name
    # mid-saturation colors that missed the bins
    return "brown" if v < 110 else "gray"


def describe_person(crop: np.ndarray, frame_h: int | None = None) -> dict | None:
    """Describe a person crop: dominant clothing colors + height class.

    The torso band (25%..80% of the crop height) is quantized in HSV; the
    top 3 colors by pixel share are reported. Returns None for degenerate
    crops. ``frame_h`` enables the apparent-height class; without it the
    height fields are null.
    """
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 24 or w < 12:
        return None
    torso = crop[int(h * 0.25):int(h * 0.80), :, :]
    if torso.size == 0:
        return None
    hsv = np.array(cv2.cvtColor(torso, cv2.COLOR_BGR2HSV), dtype=np.int16)
    hs, ss, vs = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    names = np.empty(hs.shape, dtype=object)
    for (lo, hi), smin, vmin, name in _HSV_BINS:
        m = ((hs >= lo) & (hs <= hi) & (ss >= smin) & (vs >= vmin))
        names[m] = name
    # gray/black/white fill for pixels that matched no hue bin
    unmatched = names == None  # noqa: E711  (object array comparison)
    for idx in np.argwhere(unmatched):
        y, x = int(idx[0]), int(idx[1])
        names[y, x] = _color_name(int(hs[y, x]), int(ss[y, x]), int(vs[y, x]))
    unique, counts = np.unique(names, return_counts=True)
    order = np.argsort(-counts)
    colors = [{"name": str(unique[i]), "fraction": round(
        float(counts[i]) / counts.sum(), 3)} for i in order[:3]]
    out = {"colors": colors, "height_class": None, "height_norm": None}
    if frame_h and frame_h > 0:
        hnorm = round(float(h) / float(frame_h), 3)
        out["height_norm"] = hnorm
        if hnorm >= 0.60:
            out["height_class"] = "tall"
        elif hnorm <= 0.35:
            out["height_class"] = "short"
        else:
            out["height_class"] = "medium"
    return out


def match(desc: dict | None, colors: list[str] | None = None,
          height: str | None = None) -> float:
    """Score a description against a query, 0..1 (0 = no match).

    Color score = share of the best query color present in the torso
    palette; height score = 1 when the class agrees (or no height was
    asked). Colors dominate (0.75) so a red shirt beats a height match.
    """
    if not desc:
        return 0.0
    color_score = 1.0
    if colors:
        palette = {c["name"] for c in desc.get("colors", [])}
        best = 0.0
        for c in colors:
            c = (c or "").strip().lower()
            if c and c in palette:
                frac = next((x["fraction"] for x in desc["colors"]
                             if x["name"] == c), 0.0)
                best = max(best, frac)
        if best <= 0.0:
            return 0.0
        color_score = 0.4 + 0.6 * best  # present but small share still counts
    height_score = 1.0
    if height:
        height = height.strip().lower()
        height_score = 1.0 if desc.get("height_class") == height else 0.0
    return round(0.75 * color_score + 0.25 * height_score, 3)


def search_subjects(subjects: list[dict], colors: list[str] | None = None,
                    height: str | None = None, limit: int = 20) -> list[dict]:
    """Rank gallery subjects by physical-description match."""
    scored = []
    for s in subjects:
        score = match(s.get("description"), colors, height)
        if score > 0.0:
            scored.append({"subject": s, "score": score})
    scored.sort(key=lambda r: (-r["score"], -(r["subject"].get("last_seen")
                                              or 0)))
    return scored[:max(1, limit)]


_BASE_COLORS = {"red", "orange", "yellow", "green", "cyan",
               "blue", "purple", "pink", "black", "white",
               "gray", "brown"}


_COLOR_ALIASES = {
    "navy": "blue", "dark blue": "blue", "light blue": "blue",
    "teal": "cyan", "turquoise": "cyan", "maroon": "red",
    "crimson": "red", "burgundy": "red", "scarlet": "red",
    "salmon": "pink", "magenta": "pink", "violet": "purple",
    "lavender": "purple", "lilac": "purple", "olive": "green",
    "lime": "green", "emerald": "green", "khaki": "yellow",
    "beige": "gray", "tan": "brown", "brown": "brown",
    "grey": "gray", "charcoal": "black", "dark": "black",
    "white": "white", "black": "black", "gray": "gray",
}


def parse_query(q: str) -> tuple[list[str], str | None]:
    """Split a free-text query into color names and a height class.

    ``"red shirt, tall"`` -> ``(["red"], "tall")``. Unknown words are
    ignored so casual phrasing works; height wins if multiple classes
    appear (last one wins).
    """
    colors: list[str] = []
    height: str | None = None
    for token in re.split(r"[,;]", (q or "").lower()):
        for word in token.split():
            word = word.strip(" .,!?")
            if not word:
                continue
            if word in ("tall", "medium", "short"):
                height = word
                continue
            base = _COLOR_ALIASES.get(word) or (
                word if word in _BASE_COLORS else None)
            if base and base not in colors:
                colors.append(base)
    return colors, height
