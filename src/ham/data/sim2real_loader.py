"""Sim2Real-Fire dataset loader.

Adapted from Gahtan, Shpund & Bronstein (2026),
``differentiable-eikonal-wildfire/experiments/wildfire/sim2real_loader.py``
(MIT License, https://github.com/BarakGahtan/differentiable-eikonal-wildfire).

Only ``Sim2RealFireLoader``, ``extract_arrival_times``, and
``find_ignition_point`` are retained here; the PyTorch-dependent
``CovariateAdapter`` has been omitted as HAMTools uses JAX.

Dataset folder structure expected at ``data_root``::

    data_root/
    └── 0014_00426/                     # Scene folder
        ├── Topography_Map/
        │   ├── Elevation.tif
        │   ├── Slope.tif
        │   └── Aspect.tif
        ├── Fuel_Map/
        │   ├── FBFM13.tif
        │   ├── Canopy_Cover.tif
        │   └── ...
        ├── Vegetation_Map/
        │   └── ...
        ├── Satellite_Images_Mask/
        │   ├── 0014_00001/             # Fire event
        │   │   ├── out1.jpg ... out72.jpg
        │   └── ...
        └── Weather_Data/
            ├── 0014_00001.txt
            └── ...
"""

import glob
import os

import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class Sim2RealFireLoader:
    """Load Sim2Real-Fire dataset with actual folder structure.

    Args:
        data_root: Path containing scene folders
            (e.g., ``.../simulation_data/data/``).
    """

    TOPO_FILES = ["Elevation.tif", "Slope.tif", "Aspect.tif"]
    FUEL_FILES = [
        "FBFM13.tif",
        "Canopy_Cover.tif",
        "Canopy_Height.tif",
        "Canopy_Bulk_Density.tif",
    ]
    VEG_FILES = [
        "Existing_Vegetation_Type.tif",
        "Existing_Vegetation_Cover.tif",
    ]

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.scenes = self._discover_scenes()

    def _discover_scenes(self) -> dict:
        """Find all scenes and their fire events."""
        scenes = {}
        for entry in os.listdir(self.data_root):
            scene_path = os.path.join(self.data_root, entry)
            if not os.path.isdir(scene_path):
                continue
            mask_dir = os.path.join(scene_path, "Satellite_Images_Mask")
            weather_dir = os.path.join(scene_path, "Weather_Data")
            if os.path.isdir(mask_dir) and os.path.isdir(weather_dir):
                events = []
                for event_folder in os.listdir(mask_dir):
                    event_path = os.path.join(mask_dir, event_folder)
                    if os.path.isdir(event_path):
                        weather_file = os.path.join(
                            weather_dir, f"{event_folder}.txt"
                        )
                        if os.path.exists(weather_file):
                            events.append(event_folder)
                if events:
                    scenes[entry] = {
                        "path": scene_path,
                        "events": sorted(events),
                    }
        return scenes

    def list_scenarios(self, scene_filter: list = None) -> list:
        """Return list of ``(scene_id, event_id)`` tuples.

        Args:
            scene_filter: Optional list of scene IDs to include.
                If None, returns all scenes.
        """
        scenarios = []
        for scene_id, info in self.scenes.items():
            if scene_filter is not None and scene_id not in scene_filter:
                continue
            for event_id in info["events"]:
                scenarios.append((scene_id, event_id))
        return scenarios

    def load_scenario(self, scene_id: str, event_id: str) -> dict:
        """Load a complete fire scenario.

        Returns:
            dict with keys:
                ``topography``: dict of numpy arrays (elevation, slope, aspect).
                ``fuel``: dict of numpy arrays.
                ``vegetation``: dict of numpy arrays.
                ``weather``: numpy array ``[T, 5]``.
                ``masks``: numpy array ``[T, H, W]`` boolean.
                ``timestamps``: list of hours from start.
                ``scene_id``: str.
                ``event_id``: str.
        """
        scene_path = self.scenes[scene_id]["path"]
        topo = self._load_tif_folder(
            os.path.join(scene_path, "Topography_Map"), self.TOPO_FILES
        )
        fuel = self._load_tif_folder(
            os.path.join(scene_path, "Fuel_Map"), self.FUEL_FILES
        )
        veg = self._load_tif_folder(
            os.path.join(scene_path, "Vegetation_Map"), self.VEG_FILES
        )
        mask_dir = os.path.join(
            scene_path, "Satellite_Images_Mask", event_id
        )
        weather_file = os.path.join(
            scene_path, "Weather_Data", f"{event_id}.txt"
        )
        masks, timestamps = self._load_mask_sequence(mask_dir)
        weather = self._load_weather(weather_file)
        return {
            "topography": topo,
            "fuel": fuel,
            "vegetation": veg,
            "weather": weather,
            "masks": masks,
            "timestamps": timestamps,
            "scene_id": scene_id,
            "event_id": event_id,
        }

    def _load_tif_folder(self, folder: str, file_list: list) -> dict:
        data = {}
        for filename in file_list:
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                key = filename.replace(".tif", "").lower()
                data[key] = self._load_tif(filepath)
        return data

    def _load_tif(self, filepath: str) -> np.ndarray:
        if HAS_RASTERIO:
            with rasterio.open(filepath) as src:
                return src.read(1).astype(np.float32)
        # Fallback to PIL (works for simple single-band TIFs)
        img = Image.open(filepath)
        return np.array(img, dtype=np.float32)

    def _load_mask_sequence(self, mask_dir: str) -> tuple:
        """Load sequence of fire masks.

        Returns:
            masks: ``[T, H, W]`` boolean array.
            timestamps: list of hours from start.
        """
        mask_files = glob.glob(os.path.join(mask_dir, "out*.jpg"))

        def _get_index(f):
            basename = os.path.basename(f)
            return int(basename.replace("out", "").replace(".jpg", ""))

        mask_files = sorted(mask_files, key=_get_index)
        masks = []
        for f in mask_files:
            img = Image.open(f).convert("L")
            arr = np.array(img)
            masks.append(arr > 127)
        masks = np.stack(masks, axis=0)
        timestamps = list(range(len(masks)))
        return masks, timestamps

    def _load_weather(self, filepath: str) -> np.ndarray:
        """Load weather data file.

        Columns: Year Month Day HHMM Temp Humidity Precip WindSpeed WindDir Extra

        Returns:
            weather: ``[T, 5]`` array —
            ``(temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos)``.
        """
        data = []
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    temp = float(parts[4])
                    humidity = float(parts[5])
                    wind_speed = float(parts[7])
                    wind_dir_rad = np.deg2rad(float(parts[8]))
                    data.append([
                        temp,
                        humidity,
                        wind_speed,
                        np.sin(wind_dir_rad),
                        np.cos(wind_dir_rad),
                    ])
        return np.array(data, dtype=np.float32)


def extract_arrival_times(masks: np.ndarray, timestamps: list) -> np.ndarray:
    """Convert a mask sequence to an arrival-time field.

    Args:
        masks: ``[T, H, W]`` boolean array (True = burned).
        timestamps: ``[T]`` list of times in hours.

    Returns:
        arrival_times: ``[H, W]`` float32 array — time when each pixel first
        burned; ``inf`` for pixels that never burned.
    """
    T, H, W = masks.shape
    arrival = np.full((H, W), np.inf, dtype=np.float32)
    for t_idx, t in enumerate(timestamps):
        newly_assigned = masks[t_idx] & np.isinf(arrival)
        arrival[newly_assigned] = float(t)
    return arrival


def find_ignition_point(masks: np.ndarray):
    """Find the ignition point from the first non-empty mask.

    Args:
        masks: ``[T, H, W]`` boolean array.

    Returns:
        ``(row, col)`` centroid of the first burned region, or ``None``.
    """
    for t in range(len(masks)):
        if masks[t].any():
            ys, xs = np.where(masks[t])
            return int(np.mean(ys)), int(np.mean(xs))
    return None
