import sys
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.diagnosis import CropDoctor

def main():
    parser = argparse.ArgumentParser(
        description="AI Crop Doctor End-to-End Diagnostic Pipeline CLI Runner"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "plant_village_subset" / "Potato___Late_blight" / "real_086.JPG"),
        help="Path to potato leaf photo"
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=28.6139,
        help="Field Latitude coordinate (default: 28.6139)"
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=77.2090,
        help="Field Longitude coordinate (default: 77.2090)"
    )
    parser.add_argument(
        "--heatmap-output",
        type=str,
        default=None,
        help="Optional custom output path for Grad-CAM visualization image"
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Leaf image not found at '{image_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"=== Running AI Crop Doctor Pipeline ===")
    print(f"Leaf Image: {image_path.name}")
    print(f"Location Coordinates: Lat {args.lat}, Lon {args.lon}\n")

    crop_doctor = CropDoctor()
    result = crop_doctor.diagnose(
        image_input=image_path,
        latitude=args.lat,
        longitude=args.lon,
        gradcam_output_path=args.heatmap_output
    )

    print("=== Complete Diagnostic Result Payload ===")
    print(result.to_json(indent=2))

if __name__ == "__main__":
    main()
