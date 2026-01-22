BASE_ATTRIBUTES_FILE = "data/base.json"

POLLUTION_PRESETS = {
    "angle": {
        "rotation": {"angle_range": (-45, 45)},
    },
    "lightning": {
        "lighting": {"brightness_range": (0.3, 1.8), "contrast_range": (0.5, 1.6), "add_shadow": True, "shadow_intensity": 0.7},
    },
    "blur": {
        "resolution": {"scale_factor": 0.5},
        "motion_blur": {"kernel_size": 8, "direction": "random"},
        "blur": {"blur_kernel_size": 9},
    },
    "noise": {
        "stains": {"num_stains": 5, "stain_size_range": (30, 60), "intensity_range": (0.5, 0.8)},
        "moire": {"frequency": 0.1, "intensity": 0.4},
        "noise": {"noise_intensity": 0.25},
    },
}