"""Camera Calibration — map between pixel coordinates and real-world GPS.

Provides:
- Homography matrix computation from calibration points
- pixel_to_world: pixel (x,y) -> GPS (lat, lng)
- world_to_pixel: GPS (lat, lng) -> pixel (x,y)
- Camera FOV visualization data

Usage:
    cal = CameraCalibrator("CAM-01", image_width=1920, image_height=1080)
    cal.add_calibration_point(pixel_x=960, pixel_y=540, lat=22.7179, lng=75.8519)
    cal.add_calibration_point(pixel_x=100, pixel_y=200, lat=22.7185, lng=75.8500)
    cal.add_calibration_point(pixel_x=1800, pixel_y=200, lat=22.7185, lng=75.8540)
    cal.add_calibration_point(pixel_x=960, pixel_y=100, lat=22.7190, lng=75.8519)
    cal.calibrate()
    gps = cal.pixel_to_world(500, 300)  # -> (22.7183, 75.8510)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger("bhairav.calibration")


@dataclass
class CalibrationPoint:
    """A single pixel ↔ GPS mapping point."""
    pixel_x: float
    pixel_y: float
    lat: float
    lng: float
    label: str = ""
    added_at: float = 0.0

    def __post_init__(self):
        if self.added_at == 0:
            self.added_at = time.time()


@dataclass
class CameraCalibrator:
    """Map between camera pixel coordinates and real-world GPS.

    Uses a perspective transformation (homography) computed from
    calibration points. Requires minimum 4 points for a full homography.

    For fewer points, falls back to an affine approximation.
    """
    camera_id: str
    image_width: int = 1920
    image_height: int = 1080
    _points: list = field(default_factory=list)
    _homography: list = field(default_factory=list, repr=False)
    _calibrated: bool = False
    _fov_degrees: float = 0.0

    def add_calibration_point(self, pixel_x: float, pixel_y: float,
                              lat: float, lng: float, label: str = "") -> None:
        """Add a pixel ↔ GPS calibration point."""
        self._points.append(CalibrationPoint(
            pixel_x=pixel_x, pixel_y=pixel_y,
            lat=lat, lng=lng, label=label,
        ))
        self._calibrated = False
        log.info("Calibration point added for %s: (%.0f,%.0f) -> (%.6f,%.6f)",
                 self.camera_id, pixel_x, pixel_y, lat, lng)

    def calibrate(self) -> bool:
        """Compute the homography matrix from calibration points.

        Uses DLT (Direct Linear Transform) for perspective mapping.
        Returns True if calibration succeeded (need >= 4 points).
        """
        if len(self._points) < 4:
            log.warning("Need at least 4 calibration points, have %d", len(self._points))
            return False

        try:
            import numpy as np

            # Build pixel <-> GPS correspondence arrays
            pixel_pts = np.array([[p.pixel_x, p.pixel_y] for p in self._points], dtype=float)
            gps_pts = np.array([[p.lat, p.lng] for p in self._points], dtype=float)

            # Use numpy polyfit for a simple perspective transform
            # Solve: lat = f(px, py), lng = g(px, py)
            # with bilinear interpolation (4+ points)
            n = len(self._points)

            if n == 4:
                # Exact solution with 4 points: simple bilinear
                # Sort points by position for consistent mapping
                xs = pixel_pts[:, 0]
                ys = pixel_pts[:, 1]

                # Build the bilinear system: val = a*x + b*y + c*x*y + d
                A = np.column_stack([xs, ys, xs * ys, np.ones(n)])
                h_lat, _, _, _ = np.linalg.lstsq(A, gps_pts[:, 0], rcond=None)
                h_lng, _, _, _ = np.linalg.lstsq(A, gps_pts[:, 1], rcond=None)
                self._homography = {'type': 'bilinear', 'h_lat': h_lat.tolist(), 'h_lng': h_lng.tolist()}
            else:
                # Overdetermined: use least squares with polynomial features
                xs = pixel_pts[:, 0]
                ys = pixel_pts[:, 1]
                # Features: x, y, x^2, y^2, xy, 1
                A = np.column_stack([xs, ys, xs**2, ys**2, xs * ys, np.ones(n)])
                h_lat, _, _, _ = np.linalg.lstsq(A, gps_pts[:, 0], rcond=None)
                h_lng, _, _, _ = np.linalg.lstsq(A, gps_pts[:, 1], rcond=None)
                self._homography = {'type': 'polynomial', 'h_lat': h_lat.tolist(), 'h_lng': h_lng.tolist()}

            self._calibrated = True
            self._estimate_fov()
            log.info("Camera %s calibrated with %d points (FOV: ~%.0f deg)",
                     self.camera_id, n, self._fov_degrees)
            return True

        except ImportError:
            log.warning("numpy required for calibration")
            return False
        except Exception as exc:
            log.error("Calibration failed: %s", exc)
            return False

        # Build the homography matrix using least squares
        # Source: pixel coordinates, Destination: GPS coordinates
        n = len(self._points)
        A = []
        b_lat = []
        b_lng = []

        for p in self._points:
            A.append([p.pixel_x, p.pixel_y, 1, 0, 0, 0,
                      -p.pixel_x * p.lat, -p.pixel_y * p.lat])
            b_lat.append(p.lat)
            A.append([0, 0, 0, p.pixel_x, p.pixel_y, 1,
                      -p.pixel_x * p.lng, -p.pixel_y * p.lng])
            b_lng.append(p.lng)

        try:
            import numpy as np
            A_np = np.array(A, dtype=float)
            b_lat_np = np.array(b_lat, dtype=float)
            b_lng_np = np.array(b_lng, dtype=float)

            # Solve for latitude mapping
            h_lat, _, _, _ = np.linalg.lstsq(A_np, b_lat_np, rcond=None)
            # Solve for longitude mapping
            h_lng, _, _, _ = np.linalg.lstsq(A_np, b_lng_np, rcond=None)

            self._homography = [h_lat.tolist(), h_lng.tolist()]
            self._calibrated = True

            # Estimate FOV from calibration points
            self._estimate_fov()

            log.info("Camera %s calibrated with %d points (FOV: ~%.0f°)",
                     self.camera_id, n, self._fov_degrees)
            return True

        except ImportError:
            log.warning("numpy required for calibration")
            return False
        except Exception as exc:
            log.error("Calibration failed: %s", exc)
            return False

    def _estimate_fov(self):
        """Estimate camera FOV from calibration point spread."""
        if len(self._points) < 2:
            return
        lats = [p.lat for p in self._points]
        lngs = [p.lng for p in self._points]
        lat_span = max(lats) - min(lats)
        lng_span = max(lngs) - min(lngs)
        # Rough FOV estimation from GPS spread
        self._fov_degrees = round(
            max(lat_span, lng_span) * 111, 1  # 1 degree lat ≈ 111 km
        )

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float] | None:
        """Convert pixel coordinates to GPS (lat, lng).

        Returns (lat, lng) or None if not calibrated.
        """
        if not self._calibrated or not self._homography:
            return None

        h = self._homography
        h_lat = h["h_lat"]
        h_lng = h["h_lng"]

        if h.get("type") == "bilinear":
            # val = a*x + b*y + c*x*y + d
            lat = h_lat[0]*px + h_lat[1]*py + h_lat[2]*px*py + h_lat[3]
            lng = h_lng[0]*px + h_lng[1]*py + h_lng[2]*px*py + h_lng[3]
        else:
            # val = a*x + b*y + c*x^2 + d*y^2 + e*xy + f
            lat = h_lat[0]*px + h_lat[1]*py + h_lat[2]*px**2 + h_lat[3]*py**2 + h_lat[4]*px*py + h_lat[5]
            lng = h_lng[0]*px + h_lng[1]*py + h_lng[2]*px**2 + h_lng[3]*py**2 + h_lng[4]*px*py + h_lng[5]

        return (round(lat, 6), round(lng, 6))

    def world_to_pixel(self, lat: float, lng: float) -> tuple[float, float] | None:
        """Convert GPS (lat, lng) to pixel coordinates using inverse bilinear interpolation.

        Returns (px, py) or None if not calibrated or point out of FOV.
        """
        if not self._calibrated or not self._points:
            return None

        # For 4 calibration points forming a quadrilateral, use inverse bilinear:
        # Find (s, t) in [0,1] x [0,1] such that GPS = bilinear interpolation
        # of the four corner GPS coords, then apply same (s,t) to pixel coords.

        if len(self._points) >= 4:
            # Use top-left, top-right, bottom-left, bottom-right
            pts = sorted(self._points, key=lambda p: (p.pixel_y, p.pixel_x))
            # Split into top row and bottom row
            top = sorted(pts[:len(pts)//2], key=lambda p: p.pixel_x)
            bot = sorted(pts[len(pts)//2:], key=lambda p: p.pixel_x)
            if len(top) >= 2 and len(bot) >= 2:
                tl, tr = top[0], top[-1]
                bl, br = bot[0], bot[-1]

                # Bilinear inverse: solve for (s, t)
                # GPS = (1-s)(1-t)*tl + s*(1-t)*tr + (1-s)*t*bl + s*t*br
                # => lat = tl.lat + s*(tr.lat - tl.lat) + t*(bl.lat - tl.lat) + s*t*(tl.lat - tr.lat - bl.lat + br.lat)
                # Same for lng
                a_lat = tl.lat; b_lat = tr.lat - tl.lat; c_lat = bl.lat - tl.lat
                d_lat = tl.lat - tr.lat - bl.lat + br.lat
                a_lng = tl.lng; b_lng = tr.lng - tl.lng; c_lng = bl.lng - tl.lng
                d_lng = tl.lng - tr.lng - bl.lng + br.lng

                # Solve numerically with Newton's method
                s, t = 0.5, 0.5
                for _ in range(20):
                    f = a_lat + b_lat*s + c_lat*t + d_lat*s*t - lat
                    g = a_lng + b_lng*s + c_lng*t + d_lng*s*t - lng
                    if abs(f) < 1e-10 and abs(g) < 1e-10:
                        break
                    # Jacobian
                    dfds = b_lat + d_lat*t
                    dfdt = c_lat + d_lat*s
                    dgds = b_lng + d_lng*t
                    dgdt = c_lng + d_lng*s
                    det = dfds * dgdt - dfdt * dgds
                    if abs(det) < 1e-12:
                        break
                    s -= (f * dgdt - g * dfdt) / det
                    t -= (g * dfds - f * dgds) / det

                if -0.1 <= s <= 1.1 and -0.1 <= t <= 1.1:
                    s = max(0, min(1, s))
                    t = max(0, min(1, t))
                    # Interpolate pixel coordinates using same (s, t)
                    px = (1-s)*(1-t)*tl.pixel_x + s*(1-t)*tr.pixel_x + (1-s)*t*bl.pixel_x + s*t*br.pixel_x
                    py = (1-s)*(1-t)*tl.pixel_y + s*(1-t)*tr.pixel_y + (1-s)*t*bl.pixel_y + s*t*br.pixel_y
                    if 0 <= px <= self.image_width and 0 <= py <= self.image_height:
                        return (round(px, 1), round(py, 1))
                return None

        # Fallback for < 4 points: use nearest-neighbor interpolation
        min_dist = float("inf")
        nearest = None
        for p in self._points:
            d = ((p.lat - lat)**2 + (p.lng - lng)**2)**0.5
            if d < min_dist:
                min_dist = d
                nearest = p
        if nearest and min_dist < 0.01:
            return (nearest.pixel_x, nearest.pixel_y)
        return None

    def get_fov_polygon(self, num_points: int = 8) -> list[dict]:
        """Get camera field-of-view as a GPS polygon (for map overlay)."""
        corners = [
            (0, 0),
            (self.image_width, 0),
            (self.image_width, self.image_height),
            (0, self.image_height),
        ]
        polygon = []
        for px, py in corners:
            gps = self.pixel_to_world(px, py)
            if gps:
                polygon.append({"lat": gps[0], "lng": gps[1]})
        return polygon

    def to_dict(self) -> dict:
        """Export calibration data for persistence."""
        return {
            "camera_id": self.camera_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "points": [
                {"pixel_x": p.pixel_x, "pixel_y": p.pixel_y,
                 "lat": p.lat, "lng": p.lng, "label": p.label}
                for p in self._points
            ],
            "calibrated": self._calibrated,
            "fov_degrees": self._fov_degrees,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraCalibrator":
        """Restore calibration from saved data."""
        cal = cls(
            camera_id=data["camera_id"],
            image_width=data.get("image_width", 1920),
            image_height=data.get("image_height", 1080),
        )
        for p in data.get("points", []):
            cal.add_calibration_point(
                p["pixel_x"], p["pixel_y"], p["lat"], p["lng"],
                label=p.get("label", ""),
            )
        if data.get("calibrated") and len(cal._points) >= 4:
            cal.calibrate()
        return cal
