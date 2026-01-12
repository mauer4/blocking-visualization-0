#!/usr/bin/env python3
"""
Generate a GIF that contrasts random output selection vs column tiling
in matrix multiplication.
"""

import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFont

DOT_RAMP = {
    "start_factor": 0.4,
    "end_factor": 0.12,
    "min_start_ms": 200,
    "min_end_ms": 80,
    "velocity_start": 2.5,
    "accel_start": 2.0,
    "accel_end": 8.0,
}

MAIN_RAMP = {
    "end_factor": 0.2,
    "min_end_ms": 80,
    "velocity_start": 3.5,
    "accel_start": 2.0,
    "accel_end": 8.0,
}

FONT_SIZES = {
    "body_px": 18,
    "section_px": 28,
    "title_px": 28,
}


def load_font(size, bold=False):
    if bold:
        candidates = [
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        candidates = [
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(font, text):
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_text_center(draw, font, center_x, y, text, fill):
    width, _ = text_size(font, text)
    draw.text((center_x - width // 2, y), text, font=font, fill=fill)


def quartic_progress(t, v0, a0, a1):
    s = 1.0 - v0 - a0 / 2.0
    a = (a1 - a0) / 6.0 - s
    b = 2.0 * s - (a1 - a0) / 6.0
    c = a0 / 2.0
    d = v0
    value = (((a * t + b) * t + c) * t + d) * t
    return min(1.0, max(0.0, value))


def accelerated_durations(start_ms, end_ms, steps, profile):
    if steps <= 1:
        return [int(round(start_ms))]
    durations = []
    for i in range(steps):
        t = i / (steps - 1)
        progress = quartic_progress(
            t,
            profile["velocity_start"],
            profile["accel_start"],
            profile["accel_end"],
        )
        duration = start_ms + (end_ms - start_ms) * progress
        durations.append(max(1, int(round(duration))))
    return durations


def draw_grid(draw, x, y, n, cell, color):
    size = n * cell
    for i in range(n + 1):
        xi = x + i * cell
        yi = y + i * cell
        draw.line((xi, y, xi, y + size), fill=color)
        draw.line((x, yi, x + size, yi), fill=color)


def draw_matrix_a(draw, x, y, n, cell, base, grid, rows, row_color, dot_cell=None, dot_color=None):
    draw.rectangle((x, y, x + n * cell, y + n * cell), fill=base)
    for r in rows:
        draw.rectangle(
            (x, y + r * cell, x + n * cell, y + (r + 1) * cell),
            fill=row_color,
        )
    if dot_cell is not None and dot_color is not None:
        r, c = dot_cell
        draw.rectangle(
            (x + c * cell, y + r * cell, x + (c + 1) * cell, y + (r + 1) * cell),
            fill=dot_color,
        )
    draw_grid(draw, x, y, n, cell, grid)


def draw_matrix_b(draw, x, y, n, cell, base, grid, cols, col_color, dot_cell=None, dot_color=None):
    draw.rectangle((x, y, x + n * cell, y + n * cell), fill=base)
    for c in cols:
        draw.rectangle(
            (x + c * cell, y, x + (c + 1) * cell, y + n * cell),
            fill=col_color,
        )
    if dot_cell is not None and dot_color is not None:
        r, c = dot_cell
        draw.rectangle(
            (x + c * cell, y + r * cell, x + (c + 1) * cell, y + (r + 1) * cell),
            fill=dot_color,
        )
    draw_grid(draw, x, y, n, cell, grid)


def draw_matrix_c(
    draw,
    x,
    y,
    n,
    cell,
    base,
    grid,
    outputs,
    output_color,
    current_output,
    current_color,
):
    draw.rectangle((x, y, x + n * cell, y + n * cell), fill=base)
    for r, c in outputs:
        draw.rectangle(
            (x + c * cell, y + r * cell, x + (c + 1) * cell, y + (r + 1) * cell),
            fill=output_color,
        )
    if current_output is not None:
        r, c = current_output
        draw.rectangle(
            (x + c * cell, y + r * cell, x + (c + 1) * cell, y + (r + 1) * cell),
            fill=current_color,
        )
    draw_grid(draw, x, y, n, cell, grid)


def elements_expr(row_count, col_count):
    total = row_count + col_count
    if total == 0:
        return "0"
    if total == 1:
        return "N"
    return f"{total}"


def build_layout(n, cell, font_body, font_title, font_section):
    grid_size = n * cell
    margin_x = 24
    margin_y = 20
    matrix_gap = 32
    section_gap = 36

    body_height = text_size(font_body, "A")[1]
    section_height = text_size(font_section, "A")[1]
    title_height = text_size(font_title, "A")[1]

    line_height = section_height + 8
    section_text_height = line_height
    matrix_label_height = body_height + 4

    title_gap = 10
    section_height = section_text_height + matrix_label_height + 8 + grid_size

    width = margin_x * 2 + grid_size * 3 + matrix_gap * 2
    height = (
        margin_y
        + title_height
        + title_gap
        + section_height
        + section_gap
        + section_height
        + margin_y
    )

    matrix_a_x = margin_x
    matrix_b_x = matrix_a_x + grid_size + matrix_gap
    matrix_c_x = matrix_b_x + grid_size + matrix_gap

    title_y = margin_y
    top_section_y = title_y + title_height + title_gap
    bottom_section_y = top_section_y + section_height + section_gap

    layout = {
        "width": width,
        "height": height,
        "grid_size": grid_size,
        "margin_x": margin_x,
        "title_y": title_y,
        "top_section_y": top_section_y,
        "bottom_section_y": bottom_section_y,
        "line_height": line_height,
        "section_text_height": section_text_height,
        "matrix_label_height": matrix_label_height,
        "matrix_a_x": matrix_a_x,
        "matrix_b_x": matrix_b_x,
        "matrix_c_x": matrix_c_x,
        "matrix_gap": matrix_gap,
    }
    return layout


def draw_section(
    draw,
    layout,
    section_y,
    title,
    count,
    elements,
    n,
    cell,
    colors,
    rows,
    cols,
    outputs,
    current_output,
    current_color,
    dot_a_cell,
    dot_b_cell,
    font_body,
    font_section,
    font_section_bold,
    bold_elements,
):
    text_y = section_y

    count_text = f""
    elements_text = f"ROWS MISSED AND NEEDED: {elements}"
    draw.text((layout["margin_x"], text_y), title, font=font_section, fill=colors["text"])
    draw_text_center(
        draw,
        font_section,
        layout["width"] // 2,
        text_y,
        count_text,
        colors["text"],
    )
    elements_font = font_section_bold if bold_elements else font_section
    elements_width, _ = text_size(elements_font, elements_text)
    elements_x = layout["width"] - layout["margin_x"] - elements_width
    draw.text((elements_x, text_y), elements_text, font=elements_font, fill=colors["text"])

    matrix_label_y = section_y + layout["section_text_height"]
    grid_size = layout["grid_size"]
    gap = layout["matrix_gap"]
    a_center = layout["matrix_a_x"] + grid_size // 2
    b_center = layout["matrix_b_x"] + grid_size // 2
    c_center = layout["matrix_c_x"] + grid_size // 2
    x_center = layout["matrix_a_x"] + grid_size + gap // 2
    eq_center = layout["matrix_b_x"] + grid_size + gap // 2

    draw_text_center(draw, font_body, a_center, matrix_label_y, "A", colors["text"])
    draw_text_center(draw, font_body, x_center, matrix_label_y, "X", colors["text"])
    draw_text_center(draw, font_body, b_center, matrix_label_y, "B", colors["text"])
    draw_text_center(draw, font_body, eq_center, matrix_label_y, "=", colors["text"])
    draw_text_center(draw, font_body, c_center, matrix_label_y, "C", colors["text"])

    matrices_y = matrix_label_y + layout["matrix_label_height"] + 6

    draw_matrix_a(
        draw,
        layout["matrix_a_x"],
        matrices_y,
        n,
        cell,
        colors["matrix"],
        colors["grid"],
        rows,
        colors["a_row"],
        dot_a_cell,
        colors["a_dot"],
    )
    draw_matrix_b(
        draw,
        layout["matrix_b_x"],
        matrices_y,
        n,
        cell,
        colors["matrix"],
        colors["grid"],
        cols,
        colors["b_col"],
        dot_b_cell,
        colors["b_dot"],
    )
    draw_matrix_c(
        draw,
        layout["matrix_c_x"],
        matrices_y,
        n,
        cell,
        colors["matrix"],
        colors["grid"],
        outputs,
        colors["output"],
        current_output,
        current_color,
    )


def generate_frames(n, cell, seed, duration_ms, font_body, font_title, font_section, font_section_bold):
    def blend_color(c0, c1, t):
        return tuple(int(round(c0[i] * (1 - t) + c1[i] * t)) for i in range(3))

    rng = random.Random(seed)
    choices = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(choices)
    block_size = n // 4
    block_area = block_size * block_size
    random_outputs = choices[:block_area]

    block_start = (n - block_size) // 2
    block_outputs = [
        (block_start + r, block_start + c)
        for r in range(block_size)
        for c in range(block_size)
    ]

    matrix_color = (255, 255, 255)
    output_current_color = (231, 111, 81)

    palette = []
    colors = {}

    def add_color(name, rgb):
        colors[name] = len(palette)
        palette.append(rgb)

    add_color("background", (246, 242, 232))
    add_color("grid", (75, 75, 75))
    add_color("matrix", matrix_color)
    add_color("a_row", (142, 202, 230))
    add_color("b_col", (244, 162, 97))
    add_color("output", (42, 157, 143))
    add_color("text", (29, 29, 29))
    add_color("a_dot", (33, 158, 188))
    add_color("b_dot", (230, 94, 49))

    shade_steps = 12
    colors["output_shades"] = []
    for idx in range(shade_steps):
        t = (idx + 1) / shade_steps
        shade = blend_color(matrix_color, output_current_color, t)
        colors["output_shades"].append(len(palette))
        palette.append(shade)
    colors["output_current"] = colors["output_shades"][-1]

    layout = build_layout(n, cell, font_body, font_title, font_section)
    palette_flat = []
    for color in palette:
        palette_flat.extend(color)
    palette_flat.extend([0] * (768 - len(palette_flat)))

    frames = []
    durations = []

    def new_canvas():
        img = Image.new("P", (layout["width"], layout["height"]), color=colors["background"])
        img.putpalette(palette_flat)
        return img, ImageDraw.Draw(img)

    def add_frame(
        count,
        top_outputs,
        bottom_outputs,
        top_rows,
        top_cols,
        bottom_rows,
        bottom_cols,
        top_current,
        bottom_current,
        top_current_color,
        bottom_current_color,
        top_dot_a,
        top_dot_b,
        bottom_dot_a,
        bottom_dot_b,
        bold_elements,
        duration,
    ):
        img, draw = new_canvas()
        draw_text_center(
            draw,
            font_title,
            layout["width"] // 2,
            layout["title_y"],
            "CHOOSE K OUTPUTS TO CALCULATE A X B = C",
            colors["text"],
        )

        top_elements = elements_expr(len(top_rows), len(top_cols))
        bottom_elements = elements_expr(len(bottom_rows), len(bottom_cols))

        draw_section(
            draw,
            layout,
            layout["top_section_y"],
            "RANDOM OUTPUTS",
            count,
            top_elements,
            n,
            cell,
            colors,
            top_rows,
            top_cols,
            top_outputs,
            top_current,
            top_current_color,
            top_dot_a,
            top_dot_b,
            font_body,
            font_section,
            font_section_bold,
            bold_elements,
        )

        draw_section(
            draw,
            layout,
            layout["bottom_section_y"],
            f"BLOCK TILING ({block_size}x{block_size})",
            count,
            bottom_elements,
            n,
            cell,
            colors,
            bottom_rows,
            bottom_cols,
            bottom_outputs,
            bottom_current,
            bottom_current_color,
            bottom_dot_a,
            bottom_dot_b,
            font_body,
            font_section,
            font_section_bold,
            bold_elements,
        )

        frames.append(img)
        durations.append(duration)

    top_first = random_outputs[0]
    bottom_first = block_outputs[0]

    add_frame(
        count=0,
        top_outputs=[],
        bottom_outputs=[],
        top_rows=set(),
        top_cols=set(),
        bottom_rows=set(),
        bottom_cols=set(),
        top_current=None,
        bottom_current=None,
        top_current_color=colors["output_current"],
        bottom_current_color=colors["output_current"],
        top_dot_a=None,
        top_dot_b=None,
        bottom_dot_a=None,
        bottom_dot_b=None,
        bold_elements=False,
        duration=2000,
    )

    dot_start = max(DOT_RAMP["min_start_ms"], int(duration_ms * DOT_RAMP["start_factor"]))
    dot_end = max(DOT_RAMP["min_end_ms"], int(duration_ms * DOT_RAMP["end_factor"]))
    dot_durations = accelerated_durations(dot_start, dot_end, n, DOT_RAMP)
    for step in range(n):
        duration = dot_durations[step]
        shade_progress = (step + 1) / n
        shade_idx = min(shade_steps - 1, int(round(shade_progress * (shade_steps - 1))))
        shade_color = colors["output_shades"][shade_idx]

        add_frame(
            count=1,
            top_outputs=[],
            bottom_outputs=[],
            top_rows={top_first[0]},
            top_cols={top_first[1]},
            bottom_rows={bottom_first[0]},
            bottom_cols={bottom_first[1]},
            top_current=top_first,
            bottom_current=bottom_first,
            top_current_color=shade_color,
            bottom_current_color=shade_color,
            top_dot_a=(top_first[0], step),
            top_dot_b=(step, top_first[1]),
            bottom_dot_a=(bottom_first[0], step),
            bottom_dot_b=(step, bottom_first[1]),
            bold_elements=False,
            duration=duration,
        )

    hold_bold = block_area == 1
    add_frame(
        count=1,
        top_outputs=[top_first],
        bottom_outputs=[bottom_first],
        top_rows={top_first[0]},
        top_cols={top_first[1]},
        bottom_rows={bottom_first[0]},
        bottom_cols={bottom_first[1]},
        top_current=top_first,
        bottom_current=bottom_first,
        top_current_color=colors["output_current"],
        bottom_current_color=colors["output_current"],
        top_dot_a=None,
        top_dot_b=None,
        bottom_dot_a=None,
        bottom_dot_b=None,
        bold_elements=hold_bold,
        duration=2000,
    )

    if block_area > 1:
        ramp_steps = max(0, block_area - 2)
        main_end = max(MAIN_RAMP["min_end_ms"], int(duration_ms * MAIN_RAMP["end_factor"]))
        main_durations = accelerated_durations(duration_ms, main_end, ramp_steps, MAIN_RAMP)
        for k in range(2, block_area + 1):
            top_outputs = random_outputs[:k]
            top_rows = {r for r, _ in top_outputs}
            top_cols = {c for _, c in top_outputs}
            top_current = top_outputs[-1]

            bottom_outputs = block_outputs[:k]
            bottom_rows = {r for r, _ in bottom_outputs}
            bottom_cols = {c for _, c in bottom_outputs}
            bottom_current = bottom_outputs[-1]

            bold_elements = k == block_area
            if k == block_area:
                duration = 2000
            else:
                duration = main_durations[k - 2]

            add_frame(
                count=k,
                top_outputs=top_outputs,
                bottom_outputs=bottom_outputs,
                top_rows=top_rows,
                top_cols=top_cols,
                bottom_rows=bottom_rows,
                bottom_cols=bottom_cols,
                top_current=top_current,
                bottom_current=bottom_current,
                top_current_color=colors["output_current"],
                bottom_current_color=colors["output_current"],
                top_dot_a=None,
                top_dot_b=None,
                bottom_dot_a=None,
                bottom_dot_b=None,
                bold_elements=bold_elements,
                duration=duration,
            )

    return frames, durations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a GIF visualizing random outputs vs column tiling."
    )
    parser.add_argument("--n", type=int, default=16, help="Matrix size N")
    parser.add_argument(
        "--cell", type=int, default=16, help="Cell size (pixels per matrix cell)"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--duration",
        type=int,
        default=800,
        help="Frame duration in milliseconds",
    )
    parser.add_argument(
        "--out",
        default="out",
        help="Output root directory (N-specific subdir will be created)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n < 4:
        raise SystemExit("N must be at least 4")
    if args.n % 4 != 0:
        raise SystemExit("N must be divisible by 4 for N/4 tiling")

    font_body = load_font(FONT_SIZES["body_px"])
    font_section = load_font(FONT_SIZES["section_px"])
    font_section_bold = load_font(FONT_SIZES["section_px"], bold=True)
    font_title = load_font(FONT_SIZES["title_px"])

    frames, durations = generate_frames(
        args.n, args.cell, args.seed, args.duration, font_body, font_title, font_section, font_section_bold
    )

    out_dir = os.path.join(args.out, f"N-{args.n}")
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, f"matmul_blocking_vs_random_N{args.n}.gif")

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    print(f"Wrote {gif_path}")


if __name__ == "__main__":
    main()
