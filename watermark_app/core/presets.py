"""Pre-configured presets for common watermark scenarios."""

PRESETS = {
    'douyin_bottom_right': {
        'name': '豆包右下角',
        'description': '豆包/抖音右下角水印',
        'mode': 'corner',
        'corner': 'bottom-right',
        'scan_pct': 0.18,
        'backend': 'lama',
        'fixed_position': (82, 94, 99, 98),
        'padding': 3,
    },
    'template_match': {
        'name': '模板匹配',
        'description': '使用参考图进行模板匹配',
        'mode': 'template',
        'backend': 'lama',
        'padding': 15,
        'scan_pct': 0.4,
    },
    'four_corner_auto': {
        'name': '四角自动检测',
        'description': '自动检测四个角落的水印',
        'mode': 'auto',
        'backend': 'lama',
        'scan_pct': 0.18,
    },
}


def get_preset(name: str) -> dict:
    """Get preset configuration by name."""
    return PRESETS.get(name, PRESETS['douyin_bottom_right'])


def list_presets() -> list:
    """List all available presets."""
    return [{'key': k, 'name': v['name'], 'description': v['description']}
            for k, v in PRESETS.items()]
