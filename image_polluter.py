"""
Image Polluter

- resolution
- stains: Add large, visible stains/spots (elliptical shapes)
- lighting: Vary lighting conditions with shadows (brightness/contrast/gradients)
- noise: Add strong noise (Gaussian + salt-and-pepper)
- blur: Apply Gaussian blur
- rotation
- motion_blur: Apply directional motion blur (horizontal/vertical/diagonal)
- moire: Add moire pattern to simulate scanning artifacts
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Literal, List, Optional
import random
import os


def reduce_resolution(
    image: np.ndarray, scale_factor: float = 0.3, interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Significantly reduce image resolution by scaling down then up.

    Args:
        image: Input image as numpy array
        scale_factor: Factor to reduce resolution (0.0-1.0) - lower means more degradation
        interpolation: Interpolation method

    Returns:
        Polluted image with reduced resolution
    """
    height, width = image.shape[:2]
    new_width = max(int(width * scale_factor), 10)  # Ensure at least 10 pixels
    new_height = max(int(height * scale_factor), 10)

    # Scale down significantly
    downscaled = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    # Scale back up to original size - use NEAREST for more pixelated effect
    upscaled = cv2.resize(downscaled, (width, height), interpolation=cv2.INTER_NEAREST)

    return upscaled


def add_stains(
    image: np.ndarray,
    num_stains: int = 5,
    stain_size_range: tuple = (40, 150),
    intensity_range: tuple = (0.4, 0.9),
) -> np.ndarray:
    """
    Add noticeable stains/spots to the image with varied shapes.

    Args:
        image: Input image as numpy array
        num_stains: Number of stains to add
        stain_size_range: (min, max) size of stains in pixels
        intensity_range: (min, max) darkness intensity (0.0-1.0)

    Returns:
        Polluted image with stains
    """
    polluted = image.copy().astype(np.float32)
    height, width = image.shape[:2]

    for _ in range(num_stains):
        # Random stain center
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)

        # Random stain size
        stain_size = random.randint(stain_size_range[0], stain_size_range[1])

        # Random intensity
        intensity = random.uniform(intensity_range[0], intensity_range[1])

        # Random stain shape (circular or elliptical)
        if random.random() < 0.5:
            # Circular stain
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= stain_size**2
        else:
            # Elliptical stain
            y, x = np.ogrid[:height, :width]
            ellipse_a = stain_size
            ellipse_b = stain_size * random.uniform(0.5, 1.5)
            angle = random.uniform(0, np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = (x - center_x) * cos_a + (y - center_y) * sin_a
            y_rot = -(x - center_x) * sin_a + (y - center_y) * cos_a
            mask = (x_rot ** 2 / ellipse_a ** 2 + y_rot ** 2 / ellipse_b ** 2) <= 1

        # Apply stain with gradient edges for realism
        if len(image.shape) == 3:
            for c in range(3):
                polluted[:, :, c][mask] = polluted[:, :, c][mask] * (1 - intensity)
        else:
            polluted[mask] = polluted[mask] * (1 - intensity)

    return np.clip(polluted, 0, 255).astype(np.uint8)


def vary_lighting(
    image: np.ndarray,
    brightness_range: tuple = (0.4, 1.6),
    contrast_range: tuple = (0.5, 1.5),
    add_shadow: bool = True,
    shadow_intensity: float = 0.5,
) -> np.ndarray:
    """
    Vary lighting conditions by adjusting brightness, contrast, and adding shadows.

    Args:
        image: Input image as numpy array
        brightness_range: (min, max) brightness multiplier
        contrast_range: (min, max) contrast multiplier
        add_shadow: Whether to add shadow effects
        shadow_intensity: Intensity of shadow (0.0-1.0)

    Returns:
        Polluted image with varied lighting and shadows
    """
    # Convert to PIL for easier enhancement
    if len(image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(image)

    # Random brightness adjustment
    brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness_factor)

    # Random contrast adjustment
    contrast_factor = random.uniform(contrast_range[0], contrast_range[1])
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_factor)

    # Convert back to OpenCV format
    if len(image.shape) == 3:
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        result = np.array(pil_image)

    # Add shadow effect
    if add_shadow:
        height, width = result.shape[:2]
        
        # Create gradient shadow from random direction
        shadow_direction = random.choice(['top', 'bottom', 'left', 'right', 'diagonal'])
        
        if shadow_direction == 'top':
            gradient = np.linspace(1 - shadow_intensity, 1, height).reshape(-1, 1)
            gradient = np.tile(gradient, (1, width))
        elif shadow_direction == 'bottom':
            gradient = np.linspace(1, 1 - shadow_intensity, height).reshape(-1, 1)
            gradient = np.tile(gradient, (1, width))
        elif shadow_direction == 'left':
            gradient = np.linspace(1 - shadow_intensity, 1, width).reshape(1, -1)
            gradient = np.tile(gradient, (height, 1))
        elif shadow_direction == 'right':
            gradient = np.linspace(1, 1 - shadow_intensity, width).reshape(1, -1)
            gradient = np.tile(gradient, (height, 1))
        else:  # diagonal
            grad_v = np.linspace(1 - shadow_intensity, 1, height).reshape(-1, 1)
            grad_h = np.linspace(1, 1 - shadow_intensity, width).reshape(1, -1)
            gradient = (grad_v + grad_h) / 2
        
        if len(result.shape) == 3:
            gradient = gradient[:, :, np.newaxis]
        
        result = (result * gradient).astype(np.uint8)

    return result


def add_noise(image: np.ndarray, noise_intensity: float = 0.2) -> np.ndarray:
    """
    Add strong random noise to the image.

    Args:
        image: Input image as numpy array
        noise_intensity: Intensity of noise (0.0-1.0)

    Returns:
        Polluted image with noise
    """
    # Gaussian noise
    noise = np.random.normal(0, noise_intensity * 255, image.shape).astype(np.float32)
    polluted = image.astype(np.float32) + noise
    
    # Add salt and pepper noise for more visible effect
    if noise_intensity > 0.1:
        sp_ratio = min(noise_intensity * 0.05, 0.05)  # Up to 5% of pixels
        height, width = image.shape[:2]
        
        # Salt (white) noise
        salt_mask = np.random.random((height, width)) < sp_ratio / 2
        if len(image.shape) == 3:
            # For color images, set all channels
            polluted[salt_mask, :] = 255
        else:
            polluted[salt_mask] = 255
        
        # Pepper (black) noise
        pepper_mask = np.random.random((height, width)) < sp_ratio / 2
        if len(image.shape) == 3:
            # For color images, set all channels
            polluted[pepper_mask, :] = 0
        else:
            polluted[pepper_mask] = 0
    
    polluted = np.clip(polluted, 0, 255).astype(np.uint8)
    return polluted


def add_moire_pattern(
    image: np.ndarray,
    frequency: float = 0.1,
    angle: float = None,
    intensity: float = 0.3,
) -> np.ndarray:
    """
    Add moire pattern effect to simulate scanning artifacts.

    Args:
        image: Input image as numpy array
        frequency: Frequency of the moire pattern
        angle: Angle of the pattern (None for random)
        intensity: Intensity of the pattern (0.0-1.0)

    Returns:
        Polluted image with moire pattern
    """
    height, width = image.shape[:2]
    
    if angle is None:
        angle = random.uniform(0, np.pi)
    
    # Create grid
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # Create pattern with rotation
    pattern = np.sin(2 * np.pi * frequency * (X * np.cos(angle) + Y * np.sin(angle)))
    
    # Normalize pattern to [0, 1]
    pattern = (pattern + 1) / 2
    
    # Apply pattern intensity
    pattern = 1 - (1 - pattern) * intensity
    
    # Apply to image
    if len(image.shape) == 3:
        pattern = pattern[:, :, np.newaxis]
    
    result = (image.astype(np.float32) * pattern).astype(np.uint8)
    return result


def apply_blur(image: np.ndarray, blur_kernel_size: int = 5) -> np.ndarray:
    """
    Apply blur to the image (Gaussian blur).

    Args:
        image: Input image as numpy array
        blur_kernel_size: Size of blur kernel (must be odd)

    Returns:
        Polluted image with blur
    """
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    return cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)


def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int = 15,
    direction: str = "random"
) -> np.ndarray:
    """
    Apply motion blur to simulate camera/document movement.

    Args:
        image: Input image as numpy array
        kernel_size: Size of motion blur kernel
        direction: Direction of blur ('horizontal', 'vertical', 'diagonal', 'random')

    Returns:
        Polluted image with motion blur
    """
    if direction == "random":
        direction = random.choice(["horizontal", "vertical", "diagonal"])
    
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    
    if direction == "horizontal":
        kernel[kernel_size // 2, :] = 1.0
    elif direction == "vertical":
        kernel[:, kernel_size // 2] = 1.0
    elif direction == "diagonal":
        np.fill_diagonal(kernel, 1.0)
    
    kernel = kernel / kernel.sum()
    
    # Apply the kernel
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def apply_rotation(
    image: np.ndarray, angle_range: tuple = (-45, 45), fill_color: tuple = (255, 255, 255)
) -> np.ndarray:
    """
    Apply significant rotation/skew to the image (10 degrees to upside down).

    Args:
        image: Input image as numpy array
        angle_range: (min, max) rotation angle in degrees
        fill_color: Color to fill background (BGR for color, grayscale value for mono)

    Returns:
        Polluted image with rotation
    """
    height, width = image.shape[:2]
    min_angle, max_angle = angle_range
    angle = random.uniform(min_angle, max_angle)

    # Get rotation matrix
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Determine fill color
    if len(image.shape) == 3:
        border_value = fill_color
    else:
        border_value = fill_color[0] if isinstance(fill_color, tuple) else fill_color

    # Apply rotation
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        borderValue=border_value,
        flags=cv2.INTER_LINEAR,
    )

    return rotated


PollutionMethod = Literal[
    "resolution", "stains", "lighting", "noise", "blur", "rotation", "motion_blur", "moire"
]


def pollute_image(
    input_path: str,
    output_path: str,
    method: PollutionMethod,
    **kwargs,
) -> str:
    """
    Apply pollution to an image using the specified method.

    Args:
        input_path: Path to input image
        output_path: Path to save polluted image
        method: Pollution method to apply
        **kwargs: Additional parameters for specific methods

    Returns:
        Path to saved polluted image

    Raises:
        ValueError: If method is not recognized
    """
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")

    # Apply pollution based on method
    if method == "resolution":
        scale_factor = kwargs.get("scale_factor", 0.5)
        polluted = reduce_resolution(image, scale_factor=scale_factor)

    elif method == "stains":
        num_stains = kwargs.get("num_stains", 3)
        stain_size_range = kwargs.get("stain_size_range", (20, 80))
        intensity_range = kwargs.get("intensity_range", (0.3, 0.7))
        polluted = add_stains(
            image,
            num_stains=num_stains,
            stain_size_range=stain_size_range,
            intensity_range=intensity_range,
        )

    elif method == "lighting":
        brightness_range = kwargs.get("brightness_range", (0.7, 1.3))
        contrast_range = kwargs.get("contrast_range", (0.8, 1.2))
        polluted = vary_lighting(
            image,
            brightness_range=brightness_range,
            contrast_range=contrast_range,
        )

    elif method == "noise":
        noise_intensity = kwargs.get("noise_intensity", 0.1)
        polluted = add_noise(image, noise_intensity=noise_intensity)

    elif method == "blur":
        blur_kernel_size = kwargs.get("blur_kernel_size", 3)
        polluted = apply_blur(image, blur_kernel_size=blur_kernel_size)

    elif method == "rotation":
        angle_range = kwargs.get("angle_range", (-45, 45))
        fill_color = kwargs.get("fill_color", (255, 255, 255))
        polluted = apply_rotation(image, angle_range=angle_range, fill_color=fill_color)

    elif method == "motion_blur":
        kernel_size = kwargs.get("kernel_size", 15)
        direction = kwargs.get("direction", "random")
        polluted = apply_motion_blur(image, kernel_size=kernel_size, direction=direction)

    elif method == "moire":
        frequency = kwargs.get("frequency", 0.1)
        angle = kwargs.get("angle", None)
        intensity = kwargs.get("intensity", 0.3)
        polluted = add_moire_pattern(image, frequency=frequency, angle=angle, intensity=intensity)

    else:
        raise ValueError(
            f"Unknown pollution method: {method}. "
            f"Available methods: resolution, stains, lighting, noise, blur, rotation, motion_blur, moire"
        )

    # Save polluted image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, polluted)

    return output_path


def pollute_image_multiple(
    input_path: str,
    output_path: str,
    methods: List[PollutionMethod],
    **kwargs,
) -> str:
    """
    Apply multiple pollution methods sequentially.

    Args:
        input_path: Path to input image
        output_path: Path to save polluted image
        methods: List of pollution methods to apply in order
        **kwargs: Parameters for each method (keyed by method name)

    Returns:
        Path to saved polluted image
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")

    polluted = image.copy()

    for method in methods:
        method_kwargs = kwargs.get(method, {})

        if method == "resolution":
            polluted = reduce_resolution(polluted, **method_kwargs)
        elif method == "stains":
            polluted = add_stains(polluted, **method_kwargs)
        elif method == "lighting":
            polluted = vary_lighting(polluted, **method_kwargs)
        elif method == "noise":
            polluted = add_noise(polluted, **method_kwargs)
        elif method == "blur":
            polluted = apply_blur(polluted, **method_kwargs)
        elif method == "rotation":
            polluted = apply_rotation(polluted, **method_kwargs)
        elif method == "motion_blur":
            polluted = apply_motion_blur(polluted, **method_kwargs)
        elif method == "moire":
            polluted = add_moire_pattern(polluted, **method_kwargs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, polluted)

    return output_path


def pollute_image_from_config(
    input_path: str,
    output_path: str,
    method: Optional[PollutionMethod] = None,
    methods: Optional[List[PollutionMethod]] = None,
    # Resolution parameters
    scale_factor: float = 0.3,
    # Stains parameters
    num_stains: int = 5,
    stain_size_min: int = 20,
    stain_size_max: int = 100,
    stain_intensity_min: float = 0.4,
    stain_intensity_max: float = 0.9,
    # Lighting parameters
    brightness_min: float = 0.4,
    brightness_max: float = 1.6,
    contrast_min: float = 0.5,
    contrast_max: float = 1.5,
    # Noise parameters
    noise_intensity: float = 0.2,
    # Blur parameters
    blur_kernel_size: int = 7,
    # Rotation parameters
    angle_min: float = -45,
    angle_max: float = 45,
    # Motion blur parameters
    motion_blur_kernel: int = 15,
    motion_blur_direction: str = "random",
    # Moire parameters
    moire_frequency: float = 0.1,
    moire_intensity: float = 0.3,
) -> str:
    """
    Apply pollution to an image based on configuration parameters.

    Args:
        input_path: Path to input image
        output_path: Path to save polluted image
        method: Single pollution method to apply (mutually exclusive with methods)
        methods: List of pollution methods to apply sequentially (mutually exclusive with method)
        scale_factor: Resolution scale factor (0.0-1.0)
        num_stains: Number of stains to add
        stain_size_min: Minimum stain size
        stain_size_max: Maximum stain size
        stain_intensity_min: Minimum stain intensity
        stain_intensity_max: Maximum stain intensity
        brightness_min: Minimum brightness
        brightness_max: Maximum brightness
        contrast_min: Minimum contrast
        contrast_max: Maximum contrast
        noise_intensity: Noise intensity (0.0-1.0)
        blur_kernel_size: Blur kernel size (odd number)
        angle_min: Minimum rotation angle
        angle_max: Maximum rotation angle

    Returns:
        Path to saved polluted image

    Raises:
        ValueError: If neither method nor methods is specified, or both are specified
    """
    if methods and method:
        raise ValueError("Cannot specify both 'method' and 'methods'")
    if not methods and not method:
        raise ValueError("Must specify either 'method' or 'methods'")

    # Prepare kwargs based on method(s)
    if methods:
        kwargs = {
            "resolution": {"scale_factor": scale_factor},
            "stains": {
                "num_stains": num_stains,
                "stain_size_range": (stain_size_min, stain_size_max),
                "intensity_range": (stain_intensity_min, stain_intensity_max),
            },
            "lighting": {
                "brightness_range": (brightness_min, brightness_max),
                "contrast_range": (contrast_min, contrast_max),
            },
            "noise": {"noise_intensity": noise_intensity},
            "blur": {"blur_kernel_size": blur_kernel_size},
            "rotation": {"angle_range": (angle_min, angle_max)},
            "motion_blur": {
                "kernel_size": motion_blur_kernel,
                "direction": motion_blur_direction,
            },
            "moire": {
                "frequency": moire_frequency,
                "intensity": moire_intensity,
            },
        }
        result = pollute_image_multiple(input_path, output_path, methods, **kwargs)
    else:
        kwargs = {}
        if method == "resolution":
            kwargs["scale_factor"] = scale_factor
        elif method == "stains":
            kwargs["num_stains"] = num_stains
            kwargs["stain_size_range"] = (stain_size_min, stain_size_max)
            kwargs["intensity_range"] = (stain_intensity_min, stain_intensity_max)
        elif method == "lighting":
            kwargs["brightness_range"] = (brightness_min, brightness_max)
            kwargs["contrast_range"] = (contrast_min, contrast_max)
        elif method == "noise":
            kwargs["noise_intensity"] = noise_intensity
        elif method == "blur":
            kwargs["blur_kernel_size"] = blur_kernel_size
        elif method == "rotation":
            kwargs["angle_range"] = (angle_min, angle_max)
        elif method == "motion_blur":
            kwargs["kernel_size"] = motion_blur_kernel
            kwargs["direction"] = motion_blur_direction
        elif method == "moire":
            kwargs["frequency"] = moire_frequency
            kwargs["intensity"] = moire_intensity

        result = pollute_image(input_path, output_path, method, **kwargs)

    return result



def main():
    """Command-line interface for image pollution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply pollution effects to images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reduce resolution (pixelation)
  python image_polluter.py input.png output.png --method resolution --scale-factor 0.2

  # Add large visible stains
  python image_polluter.py input.png output.png --method stains --num-stains 8

  # Vary lighting with shadows
  python image_polluter.py input.png output.png --method lighting

  # Apply motion blur
  python image_polluter.py input.png output.png --method motion_blur

  # Add moire pattern (scanning artifact)
  python image_polluter.py input.png output.png --method moire

  # Heavy rotation (up to 90 degrees)
  python image_polluter.py input.png output.png --method rotation --angle-min -90 --angle-max 90

  # Apply multiple methods
  python image_polluter.py input.png output.png --methods resolution motion_blur moire
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to input image",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to save polluted image",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        choices=["resolution", "stains", "lighting", "noise", "blur", "rotation", "motion_blur", "moire"],
        help="Single pollution method to apply",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["resolution", "stains", "lighting", "noise", "blur", "rotation", "motion_blur", "moire"],
        help="Multiple pollution methods to apply sequentially",
    )

    # Resolution parameters
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.5,
        help="Resolution scale factor (0.0-1.0)",
    )

    # Stains parameters
    parser.add_argument(
        "--num-stains", type=int, default=3, help="Number of stains to add"
    )
    parser.add_argument(
        "--stain-size-min", type=int, default=20, help="Minimum stain size"
    )
    parser.add_argument(
        "--stain-size-max", type=int, default=80, help="Maximum stain size"
    )
    parser.add_argument(
        "--stain-intensity-min", type=float, default=0.3, help="Minimum stain intensity"
    )
    parser.add_argument(
        "--stain-intensity-max", type=float, default=0.7, help="Maximum stain intensity"
    )

    # Lighting parameters
    parser.add_argument(
        "--brightness-min", type=float, default=0.7, help="Minimum brightness"
    )
    parser.add_argument(
        "--brightness-max", type=float, default=1.3, help="Maximum brightness"
    )
    parser.add_argument(
        "--contrast-min", type=float, default=0.8, help="Minimum contrast"
    )
    parser.add_argument(
        "--contrast-max", type=float, default=1.2, help="Maximum contrast"
    )

    # Noise parameters
    parser.add_argument(
        "--noise-intensity", type=float, default=0.1, help="Noise intensity (0.0-1.0)"
    )

    # Blur parameters
    parser.add_argument(
        "--blur-kernel-size", type=int, default=3, help="Blur kernel size (odd number)"
    )

    # Rotation parameters
    parser.add_argument(
        "--angle-min", type=float, default=-45, help="Minimum rotation angle"
    )
    parser.add_argument(
        "--angle-max", type=float, default=45, help="Maximum rotation angle"
    )

    # Motion blur parameters
    parser.add_argument(
        "--motion-blur-kernel", type=int, default=15, help="Motion blur kernel size"
    )
    parser.add_argument(
        "--motion-blur-direction", 
        type=str, 
        choices=["horizontal", "vertical", "diagonal", "random"],
        default="random", 
        help="Motion blur direction"
    )

    # Moire parameters
    parser.add_argument(
        "--moire-frequency", type=float, default=0.1, help="Moire pattern frequency"
    )
    parser.add_argument(
        "--moire-intensity", type=float, default=0.3, help="Moire pattern intensity (0.0-1.0)"
    )

    args = parser.parse_args()

    # Validate that either method or methods is specified
    if not args.methods and not args.method:
        parser.error("Must specify either --method or --methods")

    # Call the configuration-based function with parsed arguments
    result = pollute_image_from_config(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        methods=args.methods,
        scale_factor=args.scale_factor,
        num_stains=args.num_stains,
        stain_size_min=args.stain_size_min,
        stain_size_max=args.stain_size_max,
        stain_intensity_min=args.stain_intensity_min,
        stain_intensity_max=args.stain_intensity_max,
        brightness_min=args.brightness_min,
        brightness_max=args.brightness_max,
        contrast_min=args.contrast_min,
        contrast_max=args.contrast_max,
        noise_intensity=args.noise_intensity,
        blur_kernel_size=args.blur_kernel_size,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        motion_blur_kernel=args.motion_blur_kernel,
        motion_blur_direction=args.motion_blur_direction,
        moire_frequency=args.moire_frequency,
        moire_intensity=args.moire_intensity,
    )

    print(f"✓ Polluted image saved to: {result}")


if __name__ == "__main__":
    main()

"""
python batch_polluter.py --input-dir samples --output-dir polluted_samples --max-workers 6
"""
