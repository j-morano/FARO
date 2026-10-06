import re

from matplotlib.colors import to_rgb
import colorsys


MODEL_COLORS = {
    "iBOT": "#dddddd",
    "UrFound": "#17becf",
    "RETFound": "#d62728",
    "VisionFM": "#ff7f0e",
    "OCTCube": "#2ca02c",
    "MIRAGE\n(-SLO)": "#1f77b4",
    "MIRAGE\n(seg. targets)": "#e377c2",
    "MIRAGE\n(seg. targets\n+ Pat. bal.)": "#009999",
    "MIRAGEv2 =": "#9467bd",
    "CLOSER =": "#9467bd",
    "MIRAGE": "#1f77b4",
    "CLOSER (60": "#a38cb8",
    "CLOSER": "#9467bd",
    # 3D models
    "Mv2\n+XAttn\n+XAttn": "#8c564b",
    # ablation models
    "nonPB-MAE-only\n(Patch)": "#e377c2",
    # ... etc
}

def get_palette(prescribed_order):
    palette = {}
    for model_name in prescribed_order:
        color = "#7f7f7f"  # default grey
        for model_col, model_color in MODEL_COLORS.items():
            if model_col in model_name:
                color = model_color
                break
        palette[model_name] = color
    return palette


LESION_CLASSES = {
    'Cyst', 'PED', 'SRF', 'Fluid', 'IRF', 'PED', 'SHRM', 'Fluid/cyst',
    'SRM', 'ERM', 'SES', 'HTD', 'FLU', 'HRM', 'Macular hole',
}


def get_model_colors(prescribed_order, darken_factor=0.5, yellow_darken_factor=0.35,
                      closer_brighten_factor=1.05, closer_name='CLOSER'):
    """
    Build LaTeX colors matching the plot palette, one per model.
    - Most models: darkened by `darken_factor` for text readability.
    - Yellow-ish colors: darkened further (`yellow_darken_factor`) and
      slightly desaturated-boosted, since dark yellow otherwise looks muddy.
    - The CLOSER model: brightened instead of darkened, so it pops.

    Returns (color_defs, model_to_colorname):
        color_defs: list of '\\definecolor{...}{rgb}{r,g,b}' lines
        model_to_colorname: {model_label (no \n): latex color name}
    """
    palette = get_palette(prescribed_order)
    if isinstance(palette, dict):
        colors = [palette[model_name] for model_name in prescribed_order]
    else:
        colors = list(palette)

    color_defs = []
    model_to_colorname = {}
    for model_name, color in zip(prescribed_order, colors):
        model_label = model_name.replace('\n', ' ')
        r, g, b = to_rgb(color)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        is_yellow = 0.11 <= h <= 0.19  # roughly yellow/gold hue range
        is_closer = closer_name.lower() in model_label.lower()

        if is_closer:
            # Brighten: boost value, keep/slightly boost saturation
            v = min(1.0, v * closer_brighten_factor)
            s = min(1.0, s * 1.05)
        elif is_yellow:
            v = v * yellow_darken_factor
            s = min(1.0, s * 1.15)  # keep it looking yellow, not gray/brown
        else:
            v = v * darken_factor

        r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)

        color_name = 'col' + re.sub(r'[^A-Za-z]', '', model_label)
        color_defs.append(f'\\definecolor{{{color_name}}}{{rgb}}{{{r2:.3f},{g2:.3f},{b2:.3f}}}')
        model_to_colorname[model_label] = color_name
    return color_defs, model_to_colorname
