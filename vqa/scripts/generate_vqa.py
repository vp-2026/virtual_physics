"""Recovered static-question helpers. See vqa/README.md for provenance.

Helper function bodies are unchanged from the archived generator. The legacy
showcase-image generator and private default paths are intentionally omitted.
"""
from __future__ import annotations
import hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
DEFAULT_MODELS = ("gpt", "gemini", "qwen")
RELATION_ALIGNMENT_TOLERANCE_PX = 30.0

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def visible_objects(world: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        obj
        for name, obj in world["world"]["objects"].items()
        if not str(name).startswith("_")
    ]

def numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def color_of(obj: dict[str, Any]) -> str:
    return str(obj.get("color") or obj.get("innerColor") or "black").lower()

def centroid(obj: dict[str, Any]) -> tuple[float, float]:
    if "position" in obj:
        return float(obj["position"][0]), float(obj["position"][1])
    points = obj.get("points") or obj.get("vertices")
    if points:
        return (
            sum(float(point[0]) for point in points) / len(points),
            sum(float(point[1]) for point in points) / len(points),
        )
    polygons = obj.get("polygons") or obj.get("polylist") or obj.get("polys")
    if polygons:
        flat = [point for polygon in polygons for point in polygon]
        return (
            sum(float(point[0]) for point in flat) / len(flat),
            sum(float(point[1]) for point in flat) / len(flat),
        )
    raise ValueError(f"Cannot compute centroid for {obj.get('name')}")

def role_objects(objects: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    red = [obj for obj in objects if color_of(obj) == "red" and obj.get("type") == "Ball"]
    green = [obj for obj in objects if obj.get("type") == "Container" or color_of(obj) == "green"]
    blue = [obj for obj in objects if color_of(obj) == "blue"]
    if len(red) != 1 or len(green) != 1:
        raise ValueError(f"Expected one red ball and one green goal; found {len(red)}, {len(green)}")
    return red[0], green[0], blue

def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
        previous = current
    return inside

def quadrant_label(x: float, y: float, width: float, height: float) -> str:
    horizontal = "left" if x < width / 2 else "right"
    vertical = "lower" if y < height / 2 else "upper"
    return f"{vertical}-{horizontal}"

def object_points(obj: dict[str, Any]) -> list[list[float]]:
    points = obj.get("points") or obj.get("vertices")
    if points:
        return points
    polygons = obj.get("polygons") or obj.get("polylist") or obj.get("polys")
    return [point for polygon in (polygons or []) for point in polygon]

def object_bounds(obj: dict[str, Any]) -> tuple[float, float, float]:
    if obj.get("type") == "Ball":
        radius = numeric(obj.get("radius"))
        return 2 * radius, 2 * radius, math.pi * radius * radius
    points = object_points(obj)
    if not points:
        return 0.0, 0.0, 0.0
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return width, height, width * height

def blue_shape_description(obj: dict[str, Any]) -> str:
    obj_type = str(obj.get("type") or "")
    width, height, _ = object_bounds(obj)
    if obj_type == "Compound":
        return "blue U-shaped cradle"
    if obj_type == "Ball":
        return "blue ball"
    points = object_points(obj)
    unique_x = {round(float(point[0]), 4) for point in points}
    unique_y = {round(float(point[1]), 4) for point in points}
    axis_aligned_rectangle = len(points) == 4 and len(unique_x) == 2 and len(unique_y) == 2
    if axis_aligned_rectangle:
        ratio = width / max(height, 1e-9)
        if 0.8 <= ratio <= 1.25:
            return "blue square"
        if ratio >= 3:
            return "long horizontal blue rectangle"
        if ratio > 1.25:
            return "wide blue rectangle"
        if ratio <= 1 / 3:
            return "tall narrow blue rectangle"
        return "tall blue rectangle"
    if len(points) == 3:
        return "blue triangle"
    return "irregular blue polygon"

def blue_object_labels(blue: list[dict[str, Any]]) -> dict[int, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in blue:
        grouped[blue_shape_description(obj)].append(obj)
    labels: dict[int, str] = {}
    for description, objects in grouped.items():
        if len(objects) == 1:
            labels[id(objects[0])] = description
            continue
        ranked_by_area = sorted(objects, key=lambda obj: (object_bounds(obj)[2], centroid(obj)[0]))
        areas = [object_bounds(obj)[2] for obj in ranked_by_area]
        if areas[-1] > 1.2 * max(areas[0], 1e-9) and len(objects) == 2:
            labels[id(ranked_by_area[0])] = f"smaller {description}"
            labels[id(ranked_by_area[1])] = f"larger {description}"
            continue
        ranked_by_x = sorted(objects, key=lambda obj: centroid(obj)[0])
        for index, obj in enumerate(ranked_by_x):
            if len(objects) == 2:
                qualifier = "left" if index == 0 else "right"
            else:
                qualifier = f"from-left-{index + 1}"
            labels[id(obj)] = f"{qualifier} {description}"
    return labels

def sanitized_scene(world: dict[str, Any]) -> dict[str, Any]:
    dims = world["world"]["dims"]
    objects = visible_objects(world)
    red, green, blue = role_objects(objects)
    blue_labels = blue_object_labels(blue)
    black_index = 0
    output_objects = []
    for obj in objects:
        color = color_of(obj)
        if obj is red:
            label = "red target ball"
        elif obj is green:
            label = "green goal container"
        elif color == "blue":
            label = blue_labels[id(obj)]
        else:
            black_index += 1
            label = f"black fixed shape {black_index}"
        geometry = {"type": obj.get("type")}
        for key in ("position", "radius", "points", "vertices", "polygons", "polylist", "polys", "width"):
            if key in obj:
                geometry[key] = obj[key]
        output_objects.append(
            {
                "label": label,
                "color": color,
                "movable": numeric(obj.get("density")) > 0,
                "geometry": geometry,
            }
        )
    return {
        "canvas": {
            "width": dims[0],
            "height": dims[1],
            "coordinate_origin": "bottom-left",
        },
        "visible_objects": output_objects,
        "note": "This is the initial static scene. No action or motion has occurred.",
    }

def place_answer(
    question_id: str,
    prompt: str,
    options: list[str],
    answer: str,
    target_index: int,
    category: str,
) -> dict[str, Any]:
    if answer not in options:
        raise ValueError(f"Answer {answer!r} is not in options {options!r}")
    remaining = [option for option in options if option != answer]
    rotated = remaining[target_index % len(remaining) :] + remaining[: target_index % len(remaining)]
    ordered = rotated[:]
    ordered.insert(target_index % len(options), answer)
    letters = "ABCDEFGHI"
    return {
        "question_id": question_id,
        "category": category,
        "question": prompt,
        "options": {letters[index]: option for index, option in enumerate(ordered)},
        "answer": letters[ordered.index(answer)],
        "answer_value": answer,
    }

def relation(a: float, b: float, low: str, high: str, aligned: str) -> str:
    if abs(a - b) <= RELATION_ALIGNMENT_TOLERANCE_PX:
        return aligned
    return low if a < b else high

def questions_for_scene(
    layout_id: str,
    scene_index: int,
    world: dict[str, Any],
) -> list[dict[str, Any]]:
    dims = world["world"]["dims"]
    objects = visible_objects(world)
    red, green, blue = role_objects(objects)
    blue_labels = blue_object_labels(blue)
    red_xy, green_xy = centroid(red), centroid(green)
    colored_count = sum(color_of(obj) in {"red", "green", "blue"} for obj in objects)
    movable_count = sum(numeric(obj.get("density")) > 0 for obj in objects)
    inside = point_in_polygon(red_xy, green.get("points", []))
    inside_wording = scene_index % 2 == 0
    containment_answer = ("yes" if inside else "no") if inside_wording else ("no" if inside else "yes")
    containment_question = (
        "Is the red target ball's center initially inside the green goal container?"
        if inside_wording
        else "Is the red target ball's center initially outside the green goal container?"
    )
    specs = [
        ("red_shape", "What is the shape of the red target?", ["circle", "rectangle", "container", "irregular polygon"], "circle", "shape"),
        ("goal_shape", "Which description best matches the green goal?", ["open container", "solid circle", "solid triangle", "solid rectangle"], "open container", "shape"),
        ("blue_count", "How many blue objects are visible?", ["0", "1", "2", "3"], str(len(blue)), "count_color"),
        (
            "colored_count",
            "How many colored objects are visible in total, counting red, green, and blue but not black?",
            [str(value) for value in range(1, 7)],
            str(colored_count),
            "count_color",
        ),
        (
            "movable_count",
            "Using the supplied movability information, how many visible objects are movable?",
            [str(value) for value in range(1, 7)],
            str(movable_count),
            "movability",
        ),
        (
            "horizontal_relation",
            "Where is the red target ball's center horizontally relative to the green goal container's center?",
            ["left", "right", "approximately aligned"],
            relation(red_xy[0], green_xy[0], "left", "right", "approximately aligned"),
            "relative_position",
        ),
        (
            "vertical_relation",
            "Where is the red target ball's center vertically relative to the green goal container's center?",
            ["below", "above", "approximately aligned"],
            relation(red_xy[1], green_xy[1], "below", "above", "approximately aligned"),
            "relative_position",
        ),
        (
            "red_quadrant",
            "Using x=300 and y=300 as dividing lines (0-299 is left/lower; 300-599 is right/upper), "
            "which 2-by-2 quadrant contains the red target ball's center?",
            ["upper-left", "upper-right", "lower-left", "lower-right"],
            quadrant_label(red_xy[0], red_xy[1], dims[0], dims[1]),
            "coarse_location",
        ),
        (
            "goal_quadrant",
            "Using x=300 and y=300 as dividing lines (0-299 is left/lower; 300-599 is right/upper), "
            "which 2-by-2 quadrant contains the green goal container's center?",
            ["upper-left", "upper-right", "lower-left", "lower-right"],
            quadrant_label(green_xy[0], green_xy[1], dims[0], dims[1]),
            "coarse_location",
        ),
        (
            "containment",
            containment_question,
            ["yes", "no"],
            containment_answer,
            "containment",
        ),
    ]
    if len(blue) >= 2:
        first, second = sorted(blue, key=lambda obj: centroid(obj)[0])[:2]
        specs.append(
            (
                "movable_object_relation",
                f"Where is the center of the {blue_labels[id(first)]} vertically relative to "
                f"the center of the {blue_labels[id(second)]}?",
                ["below", "above", "approximately aligned"],
                relation(
                    centroid(first)[1],
                    centroid(second)[1],
                    "below",
                    "above",
                    "approximately aligned",
                ),
                "movable_object_relation",
            )
        )
    elif len(blue) == 1:
        only_blue = blue[0]
        specs.append(
            (
                "movable_object_relation",
                f"Where is the center of the {blue_labels[id(only_blue)]} horizontally relative to "
                "the red target ball's center?",
                ["left", "right", "approximately aligned"],
                relation(
                    centroid(only_blue)[0],
                    red_xy[0],
                    "left",
                    "right",
                    "approximately aligned",
                ),
                "movable_object_relation",
            )
        )
    else:
        combined_horizontal = relation(red_xy[0], green_xy[0], "left", "right", "aligned")
        combined_vertical = relation(red_xy[1], green_xy[1], "below", "above", "aligned")
        if combined_horizontal == "aligned" and combined_vertical == "aligned":
            combined_answer = "approximately the same position"
        elif combined_horizontal == "aligned":
            combined_answer = combined_vertical
        elif combined_vertical == "aligned":
            combined_answer = combined_horizontal
        else:
            combined_answer = f"{combined_vertical}-{combined_horizontal}"
        specs.append(
            (
                "movable_object_relation",
                "Which combined relation best describes the red target ball's center relative to "
                "the green goal container's center?",
                [
                    "above-left", "above-right", "below-left", "below-right",
                    "above", "below", "left", "right", "approximately the same position",
                ],
                combined_answer,
                "movable_object_relation",
            )
        )
    questions = []
    for question_index, (suffix, prompt, options, answer, category) in enumerate(specs):
        target_index = (scene_index * len(specs) + question_index) % len(options)
        questions.append(
            place_answer(
                f"{layout_id}:{suffix}",
                prompt,
                options,
                answer,
                target_index,
                category,
            )
        )
    return questions

def system_prompt() -> str:
    return (
        "You are answering perception questions about one INITIAL STATIC 2D scene. "
        "No action has been taken and no physics prediction is requested. Use only the supplied scene representation. "
        "Return one JSON object mapping every question_id to exactly one option letter."
    )

def build_user_prompt(questions: list[dict[str, Any]], representation_note: str, green_movable: bool) -> str:
    lines = [
        representation_note,
        "Scene conventions: the red target and blue filled objects are movable; black shapes are fixed. "
        f"The green goal container is {'movable' if green_movable else 'fixed'}.",
        "Answer all questions. Do not infer any future motion.",
        "",
    ]
    for question in questions:
        lines.append(f"{question['question_id']}: {question['question']}")
        lines.extend(f"  {letter}. {value}" for letter, value in question["options"].items())
    return "\n".join(lines)
