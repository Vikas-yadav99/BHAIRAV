"""3D scene data manager for real-time visualization.

Maintains a spatial model of cameras, tracked persons, and zones
that the frontend Three.js scene renders.  Updates come from the
per-frame callback; the frontend polls or receives via WebSocket.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class Camera3D:
    """3D camera position and FOV for scene rendering."""
    camera_id: str
    name: str
    x: float = 0.0
    y: float = 1.5          # height (meters)
    z: float = 0.0
    rotation_y: float = 0.0  # yaw in degrees
    rotation_x: float = -15.0  # pitch (negative = looking down)
    fov: float = 60.0
    stream_url: str = ""
    online: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.camera_id, "name": self.name,
            "position": [self.x, self.y, self.z],
            "rotation": [self.rotation_x, self.rotation_y, 0],
            "fov": self.fov, "stream_url": self.stream_url,
            "online": self.online,
        }


@dataclass
class Person3D:
    """A tracked person projected into 3D world space."""
    track_id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    bbox_2d: list = field(default_factory=list)  # [x1,y1,x2,y2] normalised
    label: str = "person"
    zone: str = ""
    camera_id: str = ""
    reid_id: str = ""
    pose: str = ""  # standing / sitting / fallen
    alert: str = ""

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "position": [self.x, self.y, self.z],
            "bbox_2d": self.bbox_2d,
            "label": self.label, "zone": self.zone,
            "camera_id": self.camera_id, "reid_id": self.reid_id,
            "pose": self.pose, "alert": self.alert,
        }


@dataclass
class Zone3D:
    """A 3D zone polygon on the ground plane."""
    name: str
    kind: str = "restricted"  # restricted / entry / public
    points: list = field(default_factory=list)  # [[x,z], ...] normalised
    color: str = "#ff000044"

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "points": self.points, "color": self.color}


class Scene3DManager:
    """Manages the 3D scene state.

    Parameters
    ----------
    world_size : tuple[float, float]
        Width x depth of the ground plane in meters (default 100x100).
    """

    def __init__(self, world_size: tuple[float, float] = (100.0, 100.0)):
        self.world_w, self.world_d = world_size
        self._cameras: dict[str, Camera3D] = {}
        self._persons: dict[int, Person3D] = {}
        self._zones: list[Zone3D] = []
        self._events: list[dict] = []  # recent events for timeline
        self._frame_count = 0

    def add_camera(self, cam: Camera3D) -> None:
        self._cameras[cam.camera_id] = cam

    def remove_camera(self, camera_id: str) -> None:
        self._cameras.pop(camera_id, None)

    def update_zone(self, zone: Zone3D) -> None:
        for i, z in enumerate(self._zones):
            if z.name == zone.name:
                self._zones[i] = zone
                return
        self._zones.append(zone)

    def update_persons(self, camera_id: str, tracks: list[dict],
                       frame_width: int = 640, frame_height: int = 480) -> None:
        """Project 2D detections into 3D world coordinates.

        Simple projection: bbox center -> world position based on
        camera position and ground-plane intersection.
        """
        cam = self._cameras.get(camera_id)
        if not cam:
            return
        seen_ids = set()
        for t in tracks:
            tid = t.get("track_id", t.get("id", 0))
            bbox = t.get("bbox", t.get("bbox_2d", [0, 0, 0, 0]))
            if len(bbox) < 4:
                continue
            # normalised center
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            # project to world: person at bottom of bbox on ground plane
            # simple perspective: lower in frame = closer
            depth = 5.0 + (1.0 - cy) * 20.0  # 5-25m range
            # lateral offset from camera center
            angle_rad = math.radians(cam.rotation_y)
            lateral = (cx - 0.5) * depth * 1.2
            wx = cam.x + math.sin(angle_rad) * depth + math.cos(angle_rad) * lateral
            wz = cam.z + math.cos(angle_rad) * depth - math.sin(angle_rad) * lateral

            zone = self._find_zone(wx, wz)
            alert = t.get("alert", "")
            pose = t.get("pose", "standing")
            if cy > 0.8:
                pose = "fallen"

            person = Person3D(
                track_id=tid, x=round(wx, 2), y=0.0, z=round(wz, 2),
                bbox_2d=[round(b, 4) for b in bbox[:4]],
                zone=zone, camera_id=camera_id,
                reid_id=str(t.get("reid_id", "")),
                pose=pose, alert=alert,
            )
            self._persons[tid] = person
            seen_ids.add(tid)

        # remove stale tracks from this camera
        for tid in list(self._persons.keys()):
            if self._persons[tid].camera_id == camera_id and tid not in seen_ids:
                del self._persons[tid]

        self._frame_count += 1

    def _find_zone(self, x: float, z: float) -> str:
        for zone in self._zones:
            if self._point_in_polygon(x, z, zone.points):
                return zone.name
        return ""

    @staticmethod
    def _point_in_polygon(x: float, z: float, polygon: list) -> bool:
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, zi = polygon[i]
            xj, zj = polygon[j]
            if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
                inside = not inside
            j = i
        return inside

    def add_event(self, event: dict) -> None:
        event.setdefault("timestamp", time.time())
        self._events.append(event)
        self._events = self._events[-500:]

    def snapshot(self) -> dict:
        return {
            "frame_count": self._frame_count,
            "world_size": [self.world_w, self.world_d],
            "cameras": [c.to_dict() for c in self._cameras.values()],
            "persons": [p.to_dict() for p in self._persons.values()],
            "zones": [z.to_dict() for z in self._zones],
            "recent_events": self._events[-50:],
        }

    def reset(self) -> None:
        self._persons.clear()
        self._events.clear()
        self._frame_count = 0
