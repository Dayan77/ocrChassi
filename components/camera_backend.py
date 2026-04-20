"""
Camera abstraction layer.

Supported backends:
  - "basler"    — Basler cameras (USB3 Vision, GigE Vision, CameraLink) via pypylon
  - "realsense" — Intel RealSense (D4xx / L5xx) color stream via pyrealsense2
  - "generic"   — Standard USB webcams via OpenCV / V4L2
  - "auto"      — Try Basler → RealSense → V4L2, picking the first one available

config_ini fields used:
  cam_type       list[str]  — backend per slot, e.g. ["basler", "realsense"]
  cam_serial     list[str]  — optional serial number per slot (empty = first found)
  cam_usb_index  list[int]  — V4L2 device index (used by "generic" only)
"""
from __future__ import annotations

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class AbstractCamera(ABC):
    """Common interface — mirrors cv2.VideoCapture for drop-in use in Thread."""

    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def isOpened(self) -> bool: ...

    @abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray]]: ...

    @abstractmethod
    def release(self): ...

    def apply_settings(self, config_idx: int):
        """Apply exposure / WB / focus from config_ini for the given slot index."""
        pass


# ---------------------------------------------------------------------------
# V4L2 helpers — identify real capture nodes vs auxiliary nodes
# ---------------------------------------------------------------------------

# VIDIOC_QUERYCAP ioctl — sizeof(struct v4l2_capability) = 104
_V4L2_QUERYCAP   = (2 << 30) | (104 << 16) | (ord('V') << 8) | 0   # 0x80685600
_V4L2_CAP_CAPTURE     = 0x00000001
_V4L2_CAP_DEVICE_CAPS = 0x80000000


def _v4l2_capture_caps(v4l2_index: int) -> Optional[int]:
    """Return device-level capability bits of /dev/videoN via VIDIOC_QUERYCAP,
    or None if the device can't be queried. Does not open a streaming session."""
    import os, fcntl, struct
    dev = f"/dev/video{v4l2_index}"
    if not os.path.exists(dev):
        return None
    try:
        fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        buf = bytearray(104)
        fcntl.ioctl(fd, _V4L2_QUERYCAP, buf)
        # capabilities @ 84, device_caps @ 88; use device_caps when available
        capabilities = struct.unpack_from('<I', buf, 84)[0]
        device_caps  = struct.unpack_from('<I', buf, 88)[0]
        return device_caps if (capabilities & _V4L2_CAP_DEVICE_CAPS) else capabilities
    except (OSError, struct.error):
        return None
    finally:
        os.close(fd)


def _is_v4l2_capture_node(v4l2_index: int) -> bool:
    """True if /dev/videoN exposes VIDEO_CAPTURE (the only type that produces frames)."""
    caps = _v4l2_capture_caps(v4l2_index)
    return caps is not None and bool(caps & _V4L2_CAP_CAPTURE)


def _list_v4l2_capture_nodes() -> list[int]:
    """Return sorted list of /dev/videoN indices that are real capture nodes."""
    import glob, os
    found: list[int] = []
    for path in glob.glob("/dev/video*"):
        name = os.path.basename(path)
        try:
            idx = int(name.replace("video", ""))
        except ValueError:
            continue
        if _is_v4l2_capture_node(idx):
            found.append(idx)
    return sorted(found)


# ---------------------------------------------------------------------------
# Generic — OpenCV / V4L2
# ---------------------------------------------------------------------------

class GenericCamera(AbstractCamera):
    """Standard OpenCV/V4L2 backend for generic USB webcams."""

    def __init__(self, v4l2_index: int):
        self._index = v4l2_index
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        # Validate that /dev/videoN is an actual capture node — UVC cameras
        # often expose multiple video nodes (capture + metadata/depth) with
        # the same name. Opening an auxiliary node succeeds but yields no frame,
        # and touching it can break streaming on sibling nodes of the same device.
        if self._index >= 0 and not _is_v4l2_capture_node(self._index):
            valid = _list_v4l2_capture_nodes()
            print(f"GenericCamera: /dev/video{self._index} não é um capture node "
                  f"(provável metadata/depth). Capture nodes disponíveis: {valid}")
            return False

        # Use default backend (V4L2 on Linux). CAP_V4L2 flag is avoided because
        # it triggers strict FOURCC negotiation that stalls older UVC cameras.
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        # CAP_PROP_TIMEOUT was renamed to CAP_PROP_READ_TIMEOUT_MSEC in OpenCV 4.x.
        # Wrap in try-except so an unsupported property never crashes open().
        try:
            prop = getattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC',
                           getattr(cv2, 'CAP_PROP_TIMEOUT', None))
            if prop is not None:
                self._cap.set(prop, 500)  # 500 ms max per read — keeps thread interruptible
        except Exception:
            pass
        return True

    def isOpened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._cap:
            return False, None
        return self._cap.read()

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def apply_settings(self, config_idx: int):
        pass


# ---------------------------------------------------------------------------
# Basler PyPylon (USB3 Vision, GigE Vision, CameraLink)
# ---------------------------------------------------------------------------

class BaslerCamera(AbstractCamera):
    """Basler PyPylon backend. Transparently supports USB3, GigE and CameraLink."""

    def __init__(self, serial: Optional[str] = None, list_index: int = 0):
        self._serial = serial or None
        self._list_index = list_index
        self._camera = None
        self._converter = None
        self._opened = False

    def open(self) -> bool:
        try:
            from pypylon import pylon
            tl_factory = pylon.TlFactory.GetInstance()

            if self._serial:
                di = pylon.DeviceInfo()
                di.SetSerialNumber(self._serial)
                matches = tl_factory.EnumerateDevices([di])
                if not matches:
                    print(f"Basler: câmera com serial '{self._serial}' não encontrada.")
                    return False
                device_info = matches[0]
            else:
                devices = tl_factory.EnumerateDevices()
                if not devices:
                    print("Basler: nenhuma câmera encontrada.")
                    return False
                idx = min(self._list_index, len(devices) - 1)
                device_info = devices[idx]

            camera = pylon.InstantCamera(tl_factory.CreateDevice(device_info))
            camera.Open()
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # Converter always outputs BGR8 regardless of camera pixel format (Bayer, Mono, YCbCr…)
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self._camera = camera
            self._converter = converter
            self._opened = True
            transport = device_info.GetDeviceClass()  # e.g. "BaslerGigE", "BaslerUsb"
            print(f"Basler: '{device_info.GetModelName()}' aberta "
                  f"(serial: {device_info.GetSerialNumber()}, transport: {transport})")
            return True
        except Exception as e:
            print(f"BaslerCamera.open() erro: {e}")
            return False

    def isOpened(self) -> bool:
        return self._opened and self._camera is not None and self._camera.IsOpen()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        try:
            from pypylon import pylon
            if not self._camera.IsGrabbing():
                return False, None
            result = self._camera.RetrieveResult(500, pylon.TimeoutHandling_Return)
            if result and result.GrabSucceeded():
                converted = self._converter.Convert(result)
                frame = converted.Array.copy()
                result.Release()
                return True, frame
            if result:
                result.Release()
            return False, None
        except Exception:
            return False, None

    def release(self):
        try:
            if self._camera:
                if self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
                self._camera.Close()
        except Exception as e:
            print(f"BaslerCamera.release() erro: {e}")
        finally:
            self._opened = False
            self._camera = None
            self._converter = None

    def apply_settings(self, config_idx: int):
        if not self._camera or not self._opened:
            return
        try:
            import config_ini
            if config_ini.cam_auto_exposure[config_idx]:
                self._camera.ExposureAuto.Value = "Continuous"
            else:
                self._camera.ExposureAuto.Value = "Off"
                # PyPylon expects microseconds; config stores ms-like values
                self._camera.ExposureTime.Value = float(abs(config_ini.cam_exposure[config_idx]) * 100)
            self._camera.GainAuto.Value = "Off"
        except Exception as e:
            print(f"BaslerCamera.apply_settings() erro: {e}")


# ---------------------------------------------------------------------------
# Intel RealSense (D4xx / L5xx via librealsense / pyrealsense2)
# ---------------------------------------------------------------------------

class RealSenseCamera(AbstractCamera):
    """Intel RealSense backend — color stream only (what OCR needs)."""

    # Preferred profile order: first match wins. Covers USB3 (30 fps) and
    # USB2 (D455 caps out at 15 fps @ 720p; 30 fps available only at ≤VGA).
    _PREFERRED_PROFILES = [
        (1280, 720, 30),
        (1280, 720, 15),
        (640, 480, 30),
        (640, 480, 15),
        (424, 240, 30),
    ]

    def __init__(self, serial: Optional[str] = None, list_index: int = 0):
        self._serial = serial or None
        self._list_index = list_index
        self._pipeline = None
        self._opened = False

    def open(self) -> bool:
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devs = list(ctx.query_devices())
            if not devs:
                print("RealSense: nenhuma câmera encontrada.")
                return False

            if self._serial:
                found = next((d for d in devs
                              if d.get_info(rs.camera_info.serial_number) == self._serial), None)
                if not found:
                    print(f"RealSense: serial '{self._serial}' não encontrado.")
                    return False
                device = found
            else:
                idx = min(self._list_index, len(devs) - 1)
                device = devs[idx]

            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)

            pipeline = rs.pipeline()
            profile = None
            last_err = None
            # enable_stream() never raises — the profile is only validated on
            # pipeline.start(). Try each preferred (w,h,fps), then wildcard.
            for attempt in list(self._PREFERRED_PROFILES) + [None]:
                config = rs.config()
                config.enable_device(serial)
                if attempt is None:
                    config.enable_stream(rs.stream.color)          # any profile
                else:
                    w, h, fps = attempt
                    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                try:
                    profile = pipeline.start(config)
                    break
                except RuntimeError as e:
                    last_err = e
                    continue
            if profile is None:
                print(f"RealSenseCamera.open() erro: nenhum perfil aceito ({last_err})")
                return False
            vs = profile.get_stream(rs.stream.color).as_video_stream_profile()

            self._pipeline = pipeline
            self._opened = True
            print(f"RealSense: '{name}' aberta (serial: {serial}, "
                  f"{vs.width()}x{vs.height()}@{vs.fps()})")
            return True
        except Exception as e:
            print(f"RealSenseCamera.open() erro: {e}")
            return False

    def isOpened(self) -> bool:
        return self._opened and self._pipeline is not None

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        try:
            # 500 ms matches the interrupt budget used by Basler.read()
            frames = self._pipeline.wait_for_frames(500)
            color = frames.get_color_frame()
            if not color:
                return False, None
            frame = np.asanyarray(color.get_data()).copy()
            return True, frame
        except Exception:
            return False, None

    def release(self):
        try:
            if self._pipeline:
                self._pipeline.stop()
        except Exception as e:
            print(f"RealSenseCamera.release() erro: {e}")
        finally:
            self._pipeline = None
            self._opened = False

    def apply_settings(self, config_idx: int):
        if not self._pipeline or not self._opened:
            return
        try:
            import pyrealsense2 as rs
            import config_ini
            sensor = self._pipeline.get_active_profile().get_device().first_color_sensor()
            if config_ini.cam_auto_exposure[config_idx]:
                sensor.set_option(rs.option.enable_auto_exposure, 1)
            else:
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure,
                                  float(abs(config_ini.cam_exposure[config_idx])))
        except Exception as e:
            print(f"RealSenseCamera.apply_settings() erro: {e}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BASLER_VENDOR_ID = "2676"  # Basler AG
_REALSENSE_VENDOR_ID = "8086"  # Intel


def _resolve_v4l2_index(raw: object) -> Optional[int]:
    """
    Resolves cam_usb_index to an integer V4L2 device index.
    Accepts:
      - int  → used directly
      - str that is a digit (e.g. "0") → converted to int
      - str camera name → scanned from /sys/class/video4linux/*/name
    Returns None if the device is not found.
    """
    import os, glob
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        if raw.isdigit():
            return int(raw)
        paths = sorted(
            glob.glob("/sys/class/video4linux/video*/name"),
            key=lambda p: int(os.path.basename(os.path.dirname(p)).replace("video", ""))
        )
        for path in paths:
            try:
                with open(path) as f:
                    if f.read().strip() == raw:
                        dev = os.path.basename(os.path.dirname(path))
                        return int(dev.replace("video", ""))
            except Exception:
                pass
    return None


def _detect_vendor(v4l2_index: int) -> str:
    """Walk sysfs to find the USB vendor ID for a V4L2 device index."""
    import os
    try:
        real = os.path.realpath(f"/sys/class/video4linux/video{v4l2_index}/device")
        path = real
        for _ in range(6):
            f = os.path.join(path, "idVendor")
            if os.path.exists(f):
                with open(f) as fh:
                    return fh.read().strip().lower()
            path = os.path.dirname(path)
    except Exception:
        pass
    return ""


class CameraFactory:
    """Creates the correct backend based on config_ini settings."""

    @staticmethod
    def create(config_idx: int) -> AbstractCamera:
        """
        Returns the right AbstractCamera for the given config slot.

        Explicit cam_type honours the user's choice — if the requested device
        (matched by serial when provided) can't be opened, the slot stays
        offline. Only "auto" chains backends with fallback.

        Reads cam_type[config_idx] from config_ini:
          "auto"      — try Basler → RealSense → V4L2, stopping at first match
          "basler"    — force Basler PyPylon; offline if not found
          "realsense" — force Intel RealSense; offline if not found
          "generic"   — force OpenCV/V4L2 at cam_usb_index[config_idx]
        """
        import config_ini

        raw_index = config_ini.cam_usb_index[config_idx] if config_idx < len(config_ini.cam_usb_index) else 0
        v4l2_index = _resolve_v4l2_index(raw_index)

        cam_types = getattr(config_ini, 'cam_type', [])
        cam_type = cam_types[config_idx] if config_idx < len(cam_types) else "auto"

        serials = getattr(config_ini, 'cam_serial', [])
        serial = (serials[config_idx] or None) if config_idx < len(serials) else None

        # --- Explicit backends: no fallback, stays offline if not found ---
        if cam_type == "basler":
            cam = BaslerCamera(serial=serial, list_index=config_idx)
            cam.open()  # BaslerCamera already prints its own error reason
            if not cam.isOpened():
                print(f"Slot {config_idx}: Basler (serial={serial or 'first'}) não disponível — offline")
            return cam

        if cam_type == "realsense":
            cam = RealSenseCamera(serial=serial, list_index=config_idx)
            cam.open()
            if not cam.isOpened():
                print(f"Slot {config_idx}: RealSense (serial={serial or 'first'}) não disponível — offline")
            return cam

        if cam_type == "generic":
            if v4l2_index is None:
                print(f"Slot {config_idx}: V4L2 '{raw_index}' não resolvido — offline")
                return GenericCamera(v4l2_index=-1)
            return GenericCamera(v4l2_index=v4l2_index)

        # --- "auto" — probe each backend in order, fall back if empty ---
        try:
            from pypylon import pylon
            if pylon.TlFactory.GetInstance().EnumerateDevices():
                cam = BaslerCamera(serial=serial, list_index=config_idx)
                if cam.open():
                    return cam
        except Exception:
            pass
        try:
            import pyrealsense2 as rs
            if list(rs.context().query_devices()):
                cam = RealSenseCamera(serial=serial, list_index=config_idx)
                if cam.open():
                    return cam
        except Exception:
            pass
        if v4l2_index is None:
            print(f"Slot {config_idx}: nenhuma câmera encontrada (auto) — offline")
            return GenericCamera(v4l2_index=-1)
        return GenericCamera(v4l2_index=v4l2_index)
