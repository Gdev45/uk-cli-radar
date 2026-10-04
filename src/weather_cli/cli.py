import os
import re
import math
import time
import threading
import concurrent.futures
from collections import OrderedDict
from io import BytesIO
from urllib.request import Request, urlopen

import boto3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import matplotlib.font_manager as fm

import cartopy.crs as ccrs

from PIL import Image
from matplotlib.widgets import Button
from botocore import UNSIGNED
from botocore.config import Config
from pyproj import CRS, Transformer


RADAR_BUCKET = "met-office-radar-obs-data"
RADAR_PREFIX = "radar"

AUTO_UPDATE_MINUTES = 1

# LightningMaps / Blitzortung live map image.
LIGHTNING_URL = (
    "https://images.lightningmaps.org/blitzortung/europe/"
    "index.php?map=uk&t={}"
)
LIGHTNING_TILE_ZOOM = 7
LIGHTNING_TILE_URL = (
    "https://images.lightningmaps.org/blitzortung/europe/"
    "tiles/{z}/{x}/{y}.png?{counter}"
)
LIGHTNING_REFRESH_SECONDS = 10
LIGHTNING_EXTENT = [-10.5, 2.5, 48.5, 60.5]

def get_retro_font():
    """Choose an installed monospace font that works on Linux and Windows."""
    candidates = [
        "DejaVu Sans Mono",
        "Liberation Mono",
        "Courier New",
        "monospace",
    ]
    available = {font.name for font in fm.fontManager.ttflist}

    for candidate in candidates:
        if candidate in available:
            return candidate

    return "monospace"


RETRO_FONT = get_retro_font()

# ---------------------------------------------------------------------------
# MAP PROJECTION
# ---------------------------------------------------------------------------
# The map used to be drawn in PlateCarree (plain lon/lat), which treats one
# degree of longitude as the same length as one degree of latitude. At UK
# latitudes that makes the country look squashed vertically / stretched
# sideways. Web Mercator keeps shapes correct (and is what the satellite
# tiles are made in), so the UK now has its real proportions.

try:
    MAP_CRS = ccrs.Mercator.GOOGLE
except AttributeError:  # very old / unusual cartopy builds
    MAP_CRS = ccrs.Mercator(
        globe=ccrs.Globe(
            ellipse=None,
            semimajor_axis=6378137,
            semiminor_axis=6378137,
            nadgrids="@null",
        ),
        min_latitude=-85.0511287798066,
        max_latitude=85.0511287798066,
    )

WEB_MERCATOR_HALF = 20037508.342789244   # half the width of the Web Mercator world (m)


def lonlat_to_map(lon, lat):
    """Convert lon/lat degrees into map (Web Mercator metre) coordinates."""
    x, y = MAP_CRS.transform_point(lon, lat, ccrs.PlateCarree())
    return float(x), float(y)


# ---------------------------------------------------------------------------
# SATELLITE BASEMAP SETTINGS
# ---------------------------------------------------------------------------

SAT_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
SAT_TILE_SIZE = 256
SAT_MIN_ZOOM = 3
SAT_MAX_ZOOM = 13
SAT_MAX_TILES = 48          # most tiles fetched/stitched for one view
SAT_MARGIN = 0.15           # extra imagery loaded around the view (fraction)
SAT_WORKERS = 6             # parallel tile downloads
SAT_CACHE_MAX = 300         # tiles kept in memory
SAT_REFRESH_MS = 400        # how often the GUI checks for new imagery
SAT_RETRY_SECONDS = 30      # wait before retrying a tile that failed
SAT_ZORDER = -1             # keep the imagery underneath everything else
SAT_CREDIT = "IMAGERY: ESRI / MAXAR"

# ---------------------------------------------------------------------------
# RADAR IMAGE RESOLUTION
# ---------------------------------------------------------------------------
# cartopy re-projects the radar only for the view it is drawn in, so the radar
# is re-drawn for the new view shortly after you stop zooming / panning.

RADAR_REGRID_SCALE = 1.5     # radar pixels per screen pixel (1.0 = one-to-one)
RADAR_REGRID_MIN = 750       # never lower than cartopy's default
RADAR_REGRID_MAX = 1800      # cap so very large windows stay quick
RADAR_SETTLE_SECONDS = 0.6   # wait this long after the view stops changing

# ---------------------------------------------------------------------------
# PORTRAIT LAYOUT SETTINGS
# ---------------------------------------------------------------------------

# Tall window (width x height in inches). At 100 dpi this is 550 x 1000 px,
# so it fits on a normal 1080p screen and is clearly portrait.
FIGURE_SIZE = (5.5, 10)

# Default view is centred on the UK. The height is fixed; the width is worked
# out from the real shape of the map area so the map itself is portrait.
DEFAULT_CENTER = (-3.3, 55.05)   # lon, lat
DEFAULT_HEIGHT_DEG = 14.5        # latitude span shown

MIN_EXTENT = [-30.0, 20.0, 35.0, 70.0]

# The same limits, converted to map metres (used for zoom / pan limits).
_min_x0, _min_y0 = lonlat_to_map(MIN_EXTENT[0], MIN_EXTENT[2])
_min_x1, _min_y1 = lonlat_to_map(MIN_EXTENT[1], MIN_EXTENT[3])
MIN_EXTENT_M = [_min_x0, _min_x1, _min_y0, _min_y1]

RAIN_VMIN = 0.05
RAIN_VMAX = 20.0

# Bottom UI stack (fractions of figure height):
#   status bar -> button row 1 -> button row 2 -> colour bar -> map
STATUS_BAR_Y = 0.005
STATUS_BAR_H = 0.05
BUTTON_COLS = 4
BUTTON_X = 0.02
BUTTON_GAP = 0.008
BUTTON_H = 0.024
BUTTON_ROW_GAP = 0.006
CBAR_Y = 0.165
CBAR_H = 0.010
MAP_BOTTOM = 0.20
MAP_TOP = 0.97


# ---------------------------------------------------------------------------
# GLOBALS USED BY THE GUI
# ---------------------------------------------------------------------------

fig = None
ax = None

update_button_ax = None
reset_button_ax = None
fax_button_ax = None
radar_button_ax = None
palette_button_ax = None
reverse_palette_button_ax = None
lightning_button_ax = None
warning_button_ax = None

update_button = None
reset_button = None
fax_button = None
radar_button = None
palette_button = None
reverse_palette_button = None
lightning_button = None
warning_button = None

status_bar_ax = None
auto_timer = None
lightning_timer = None
sat_timer = None

s3 = None
executor = None
sat_executor = None

# Satellite tile bookkeeping (shared between the GUI and download threads).
sat_cache = OrderedDict()      # (zoom, x, y) -> uint8 RGB array
sat_pending = set()            # tiles currently being downloaded
sat_failed = {}                # tile -> time of last failure
sat_lock = threading.Lock()

# Stops the timer-driven imagery refresh and a full redraw from interleaving.
draw_lock = threading.RLock()


# ---------------------------------------------------------------------------
# APPLICATION STATE
# ---------------------------------------------------------------------------

state = {
    "product": "radar",
    "key": None,
    "filename": None,
    "payload": None,
    "cbar": None,
    "cax": None,
    "status_text": None,
    "log_history": [],
    "warnings": [],
    "is_loading": False,

    "view_initialized": False,
    "is_panning": False,
    "pan_start": None,
    "pan_extent": None,

    "fax_mode": False,
    "lightning_image": None,
    "custom_radar_palette": None,
    "custom_radar_name": None,

    "sat_artist": None,
    "sat_signature": None,
    "sat_dirty": True,
    "sat_wanted": set(),
    "sat_error_logged": 0.0,
    "sat_confirmed": False,

    "radar_artist": None,
    "radar_view": None,
    "radar_style": (1.0, 0),
    "radar_busy": False,
    "view_changed_at": 0.0,
}


# ---------------------------------------------------------------------------
# STATUS LOGGING
# ---------------------------------------------------------------------------

def log_status(message):

    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"

    print(formatted_msg)

    state["log_history"].append(formatted_msg)

    if len(state["log_history"]) > 3:
        state["log_history"].pop(0)

    if state.get("status_text") is not None:
        state["status_text"].set_text("\n".join(state["log_history"]))

        if fig is not None:
            fig.canvas.draw_idle()

    normalized = str(message).upper()
    if any(token in normalized for token in ("WARNING", "WARN", "ERROR", "ALERT")):
        add_warning(message)


def add_warning(message):
    """Keep the latest warning/error messages for the warning page."""
    text = str(message).strip()
    if not text:
        return

    state["warnings"].append(text)
    if len(state["warnings"]) > 12:
        state["warnings"] = state["warnings"][-12:]

    if state["product"] == "warnings" and fig is not None:
        redraw()


# ---------------------------------------------------------------------------
# FIND LATEST RADAR
# ---------------------------------------------------------------------------

def find_latest_radar():

    log_status(
        f"S3 GET -> Querying radar prefix in 's3://{RADAR_BUCKET}'..."
    )

    now = time.gmtime()

    prefix = (
        f"{RADAR_PREFIX}/"
        f"{now.tm_year:04d}/"
        f"{now.tm_mon:02d}/"
        f"{now.tm_mday:02d}/"
    )

    objects = [
        obj
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=RADAR_BUCKET,
            Prefix=prefix
        )
        for obj in page.get("Contents", [])
        if obj["Key"].endswith("_ODIM_ng_radar_rainrate_composite_1km_UK.h5")
    ]

    if not objects:
        log_status("RADAR DATA -> No radar files found.")
        return None

    latest_key = max(objects, key=lambda obj: obj["Key"])["Key"]

    log_status(f"FOUND RADAR -> {os.path.basename(latest_key)}")

    return latest_key


# ---------------------------------------------------------------------------
# LOAD RADAR HDF5
# ---------------------------------------------------------------------------

def load_radar(filename):

    log_status(f"HDF5 READ -> Parsing radar matrix from {filename}...")

    with h5py.File(filename, "r") as f:

        data = f["dataset1/data1/data"][:]
        info = f["dataset1/data1/what"]
        where = f["where"]

        gain = float(info.attrs["gain"])
        offset = float(info.attrs["offset"])
        nodata = float(info.attrs["nodata"])

        data = data * gain + offset
        data[data == nodata] = np.nan

        projdef = where.attrs["projdef"]
        if isinstance(projdef, bytes):
            projdef = projdef.decode()

        ul_lat = float(where.attrs["UL_lat"])
        ul_lon = float(where.attrs["UL_lon"])
        lr_lat = float(where.attrs["LR_lat"])
        lr_lon = float(where.attrs["LR_lon"])

        date = f["dataset1/what"].attrs["startdate"]
        if isinstance(date, bytes):
            date = date.decode()

        starttime = f["dataset1/what"].attrs["starttime"]
        if isinstance(starttime, bytes):
            starttime = starttime.decode()

    log_status(f"PROCESSED RADAR -> Matrix shape: {data.shape}")

    return (
        data,
        projdef,
        ul_lat,
        ul_lon,
        lr_lat,
        lr_lon,
        date,
        starttime
    )


# ---------------------------------------------------------------------------
# DOWNLOAD WORKER
# ---------------------------------------------------------------------------

def download_worker(product):

    new_key = find_latest_radar()

    if new_key is None:
        return None, None, None

    filename = os.path.basename(new_key)

    log_status(f"DOWNLOAD -> Fetching {filename} from Amazon S3...")

    s3.download_file(RADAR_BUCKET, new_key, filename)

    payload = load_radar(filename)

    return new_key, filename, payload


# ---------------------------------------------------------------------------
# WINDOWS 95 BUTTON STYLE
# ---------------------------------------------------------------------------

def style_button(button_ax, button, fontsize=8):

    button_ax.set_facecolor("#c0c0c0")

    button_ax.spines["top"].set_color("#ffffff")
    button_ax.spines["left"].set_color("#ffffff")
    button_ax.spines["bottom"].set_color("#404040")
    button_ax.spines["right"].set_color("#404040")

    for side in ("top", "left", "bottom", "right"):
        button_ax.spines[side].set_linewidth(2)

    button.label.set_color("black")
    button.label.set_fontsize(fontsize)
    button.label.set_fontstyle("normal")
    button.label.set_fontweight("normal")
    button.label.set_fontname(RETRO_FONT)


# ---------------------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------------------

def set_win95_title(title_text):

    if state["fax_mode"]:
        title_color = "black"
        title_bg = "white"
        title_edge = "#444444"
    else:
        title_color = "white"
        title_bg = "black"
        title_edge = "#b89b00"

    ax.set_title(
        title_text,
        color=title_color,
        fontsize=10,
        fontweight="normal",
        fontname=RETRO_FONT,
        loc="left",
        pad=10,
        bbox=dict(
            facecolor=title_bg,
            edgecolor=title_edge,
            linewidth=1.0,
            boxstyle="square,pad=0.25"
        )
    )


# ---------------------------------------------------------------------------
# MAP VIEW
# ---------------------------------------------------------------------------
# All view extents are [x_min, x_max, y_min, y_max] in map (Web Mercator)
# metres, i.e. the axes' own coordinates. Mouse positions from Matplotlib are
# in the same units, so zoom and pan work directly on them.

def get_map_extent():
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    return [x_min, x_max, y_min, y_max]


def set_map_extent(extent):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])


def mark_view_changed():
    """Note that the view moved, so imagery and radar get refreshed."""
    state["sat_dirty"] = True
    state["view_changed_at"] = time.time()


def get_default_extent():
    """Portrait extent: fixed height, width from the map area's aspect ratio."""
    fig_w, fig_h = fig.get_size_inches() if fig is not None else FIGURE_SIZE
    area_aspect = (0.99 * fig_w) / ((MAP_TOP - MAP_BOTTOM) * fig_h)

    lon, lat = DEFAULT_CENTER

    centre_x, _ = lonlat_to_map(lon, lat)
    _, y_low = lonlat_to_map(lon, lat - DEFAULT_HEIGHT_DEG / 2)
    _, y_high = lonlat_to_map(lon, lat + DEFAULT_HEIGHT_DEG / 2)

    height = y_high - y_low
    width = height * area_aspect

    return [
        centre_x - width / 2,
        centre_x + width / 2,
        y_low,
        y_high,
    ]


def fit_extent_in_bounds(x_min, x_max, y_min, y_max):
    """Slide an extent (without resizing it) back inside MIN_EXTENT_M."""
    b_x_min, b_x_max, b_y_min, b_y_max = MIN_EXTENT_M

    if x_min < b_x_min:
        shift = b_x_min - x_min
        x_min += shift
        x_max += shift

    if x_max > b_x_max:
        shift = b_x_max - x_max
        x_min += shift
        x_max += shift

    if y_min < b_y_min:
        shift = b_y_min - y_min
        y_min += shift
        y_max += shift

    if y_max > b_y_max:
        shift = b_y_max - y_max
        y_min += shift
        y_max += shift

    return x_min, x_max, y_min, y_max


def set_default_view():

    set_map_extent(get_default_extent())

    state["view_initialized"] = True
    mark_view_changed()

    position_layout()
    fig.canvas.draw_idle()


def reset_view(event=None):

    log_status("USER EVENT -> Resetting map view...")

    state["is_panning"] = False
    state["pan_start"] = None
    state["pan_extent"] = None

    set_default_view()


# ---------------------------------------------------------------------------
# ZOOM
# ---------------------------------------------------------------------------

def zoom_map(event):

    if event.inaxes != ax:
        return

    if event.xdata is None or event.ydata is None:
        return

    button = getattr(event, "button", None)
    step = getattr(event, "step", None)

    if button in ("up", "scroll up", "scrollup", 1):
        scale = 0.75
    elif button in ("down", "scroll down", "scrolldown", -1):
        scale = 1.25
    elif step is not None:
        if step > 0:
            scale = 0.75
        elif step < 0:
            scale = 1.25
        else:
            return
    else:
        return

    x_min, x_max, y_min, y_max = get_map_extent()

    mouse_x = event.xdata
    mouse_y = event.ydata

    width = x_max - x_min
    height = y_max - y_min

    new_width = width * scale
    new_height = height * scale

    # Don't zoom out past the allowed area (keeps the map's proportions).
    max_width = MIN_EXTENT_M[1] - MIN_EXTENT_M[0]
    max_height = MIN_EXTENT_M[3] - MIN_EXTENT_M[2]

    if new_width > max_width or new_height > max_height:
        return

    rel_x = (mouse_x - x_min) / width
    rel_y = (mouse_y - y_min) / height

    new_x_min = mouse_x - rel_x * new_width
    new_x_max = mouse_x + (1.0 - rel_x) * new_width
    new_y_min = mouse_y - rel_y * new_height
    new_y_max = mouse_y + (1.0 - rel_y) * new_height

    new_x_min, new_x_max, new_y_min, new_y_max = fit_extent_in_bounds(
        new_x_min, new_x_max, new_y_min, new_y_max
    )

    if new_x_max <= new_x_min:
        return

    if new_y_max <= new_y_min:
        return

    set_map_extent([new_x_min, new_x_max, new_y_min, new_y_max])

    state["view_initialized"] = True
    mark_view_changed()

    fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# PAN
# ---------------------------------------------------------------------------

def pan_start(event):

    if event.inaxes != ax:
        return

    if event.button != 1:
        return

    if event.x is None or event.y is None:
        return

    # The drag start is stored in screen pixels (not map coordinates), because
    # map coordinates move under the mouse as the view is panned.
    state["is_panning"] = True
    state["pan_start"] = (event.x, event.y)
    state["pan_extent"] = get_map_extent()


def pan_move(event):

    if not state["is_panning"]:
        return

    if event.inaxes != ax:
        return

    if event.x is None or event.y is None:
        return

    if state["pan_start"] is None or state["pan_extent"] is None:
        return

    start_x, start_y = state["pan_start"]
    x_min, x_max, y_min, y_max = state["pan_extent"]

    box = ax.bbox

    if box.width <= 0 or box.height <= 0:
        return

    dx = (start_x - event.x) * (x_max - x_min) / box.width
    dy = (start_y - event.y) * (y_max - y_min) / box.height

    new_x_min, new_x_max, new_y_min, new_y_max = fit_extent_in_bounds(
        x_min + dx,
        x_max + dx,
        y_min + dy,
        y_max + dy,
    )

    set_map_extent([new_x_min, new_x_max, new_y_min, new_y_max])

    mark_view_changed()

    fig.canvas.draw_idle()


def pan_end(event):

    if event.button != 1:
        return

    state["is_panning"] = False
    state["pan_start"] = None
    state["pan_extent"] = None
    mark_view_changed()


# ---------------------------------------------------------------------------
# SATELLITE BASEMAP
# ---------------------------------------------------------------------------
# Imagery is Esri World Imagery in Web Mercator tiles, which match the map
# projection exactly, so the tiles are stitched into one image and placed
# straight onto the map with no re-projection.
#
# Downloads happen on background threads. The GUI only ever reads tiles that
# are already in the cache, and a timer picks up new tiles as they arrive.

def get_satellite_tile_range():
    """Work out the tile zoom and tile index range covering the current view."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return None

    pixel_width = max(float(ax.bbox.width), 200.0)
    metres_per_pixel = width / pixel_width

    full_world_res = (2.0 * WEB_MERCATOR_HALF) / SAT_TILE_SIZE
    ideal_zoom = math.log2(full_world_res / metres_per_pixel)

    zoom = int(math.ceil(ideal_zoom - 0.3))
    zoom = max(SAT_MIN_ZOOM, min(SAT_MAX_ZOOM, zoom))

    # Load a little more than is visible so small pans don't show gaps.
    x0 = max(x_min - width * SAT_MARGIN, -WEB_MERCATOR_HALF)
    x1 = min(x_max + width * SAT_MARGIN, WEB_MERCATOR_HALF)
    y0 = max(y_min - height * SAT_MARGIN, -WEB_MERCATOR_HALF)
    y1 = min(y_max + height * SAT_MARGIN, WEB_MERCATOR_HALF)

    while True:
        n = 2 ** zoom
        tile_m = (2.0 * WEB_MERCATOR_HALF) / n

        tx0 = int(math.floor((x0 + WEB_MERCATOR_HALF) / tile_m))
        tx1 = int(math.floor((x1 + WEB_MERCATOR_HALF) / tile_m))
        ty0 = int(math.floor((WEB_MERCATOR_HALF - y1) / tile_m))
        ty1 = int(math.floor((WEB_MERCATOR_HALF - y0) / tile_m))

        tx0 = max(0, min(n - 1, tx0))
        tx1 = max(0, min(n - 1, tx1))
        ty0 = max(0, min(n - 1, ty0))
        ty1 = max(0, min(n - 1, ty1))

        count = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)

        if count <= SAT_MAX_TILES or zoom <= SAT_MIN_ZOOM:
            break

        zoom -= 1

    return zoom, tx0, tx1, ty0, ty1, tile_m


def fetch_satellite_tile(key):
    """Download one imagery tile into the cache (runs on a worker thread)."""
    zoom, x, y = key

    try:
        if key not in state["sat_wanted"]:
            # The view has moved on since this was queued; skip it.
            return

        request = Request(
            SAT_TILE_URL.format(z=zoom, x=x, y=y),
            headers={"User-Agent": "UK-Retro-Radar/0.2.0"},
        )

        with urlopen(request, timeout=15) as response:
            data = response.read()

        image = Image.open(BytesIO(data)).convert("RGB")
        tile = np.asarray(image, dtype=np.uint8)

        if tile.shape != (SAT_TILE_SIZE, SAT_TILE_SIZE, 3):
            raise RuntimeError(f"Unexpected tile size {tile.shape}")

        with sat_lock:
            sat_cache[key] = tile
            sat_cache.move_to_end(key)

            while len(sat_cache) > SAT_CACHE_MAX:
                sat_cache.popitem(last=False)

            sat_failed.pop(key, None)

        state["sat_dirty"] = True

        if not state["sat_confirmed"]:
            state["sat_confirmed"] = True
            log_status("SATELLITE -> Imagery tiles loading OK.")

    except Exception as exc:
        now = time.time()

        with sat_lock:
            sat_failed[key] = now

        # Log at most once a minute so being offline doesn't flood the log.
        if now - state["sat_error_logged"] > 60:
            state["sat_error_logged"] = now
            log_status(f"SATELLITE ERROR -> {type(exc).__name__}: {exc}")

    finally:
        with sat_lock:
            sat_pending.discard(key)


def request_satellite_tiles(keys):
    """Queue downloads for any tiles that aren't cached, pending or failing."""
    if sat_executor is None:
        return

    now = time.time()
    queued = []

    with sat_lock:
        for key in keys:
            if key in sat_pending:
                continue

            if now - sat_failed.get(key, 0.0) < SAT_RETRY_SECONDS:
                continue

            sat_pending.add(key)
            queued.append(key)

    for key in queued:
        sat_executor.submit(fetch_satellite_tile, key)


def remove_satellite_layer():
    artist = state.get("sat_artist")

    if artist is not None:
        try:
            artist.remove()
        except Exception:
            pass

    state["sat_artist"] = None


def update_satellite():
    """Rebuild the satellite layer for the current view (GUI thread / redraw)."""
    if ax is None:
        return

    with draw_lock:

        state["sat_dirty"] = False

        # Fax mode is a plain white/ink display, so no imagery there.
        if state["fax_mode"]:
            remove_satellite_layer()
            state["sat_signature"] = None
            return

        tile_range = get_satellite_tile_range()

        if tile_range is None:
            return

        zoom, tx0, tx1, ty0, ty1, tile_m = tile_range

        keys = [
            (zoom, x, y)
            for y in range(ty0, ty1 + 1)
            for x in range(tx0, tx1 + 1)
        ]

        state["sat_wanted"] = set(keys)

        with sat_lock:
            missing = [key for key in keys if key not in sat_cache]

        if missing:
            request_satellite_tiles(missing)

        loaded = len(keys) - len(missing)
        signature = (zoom, tx0, tx1, ty0, ty1, loaded)

        # Nothing new to show since the last build.
        if signature == state["sat_signature"] and state["sat_artist"] is not None:
            return

        # Nothing downloaded yet: leave whatever is on screen alone.
        if loaded == 0:
            return

        cols = tx1 - tx0 + 1
        rows = ty1 - ty0 + 1

        mosaic = np.zeros(
            (rows * SAT_TILE_SIZE, cols * SAT_TILE_SIZE, 4),
            dtype=np.uint8,
        )

        with sat_lock:
            for r, ty in enumerate(range(ty0, ty1 + 1)):
                for c, tx in enumerate(range(tx0, tx1 + 1)):
                    tile = sat_cache.get((zoom, tx, ty))

                    if tile is None:
                        continue

                    sat_cache.move_to_end((zoom, tx, ty))

                    y_from = r * SAT_TILE_SIZE
                    x_from = c * SAT_TILE_SIZE

                    mosaic[
                        y_from:y_from + SAT_TILE_SIZE,
                        x_from:x_from + SAT_TILE_SIZE,
                        :3
                    ] = tile
                    mosaic[
                        y_from:y_from + SAT_TILE_SIZE,
                        x_from:x_from + SAT_TILE_SIZE,
                        3
                    ] = 255

        left = -WEB_MERCATOR_HALF + tx0 * tile_m
        right = -WEB_MERCATOR_HALF + (tx1 + 1) * tile_m
        top = WEB_MERCATOR_HALF - ty0 * tile_m
        bottom = WEB_MERCATOR_HALF - (ty1 + 1) * tile_m

        remove_satellite_layer()

        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        state["sat_artist"] = ax.imshow(
            mosaic,
            extent=[left, right, bottom, top],
            origin="upper",
            interpolation="bilinear",
            zorder=SAT_ZORDER,
        )

        # imshow can change the view; put it back exactly as it was.
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        state["sat_signature"] = signature


def draw_basemap():
    """Satellite imagery under the radar / lightning / warnings pages.

    Called right after ax.clear(), which wipes the previous imagery layer.
    """
    state["sat_artist"] = None
    state["sat_signature"] = None
    state["radar_artist"] = None
    state["radar_view"] = None

    update_satellite()

    if not state["fax_mode"]:
        ax.text(
            0.99,
            0.008,
            SAT_CREDIT,
            transform=ax.transAxes,
            color="white",
            fontsize=6,
            fontname=RETRO_FONT,
            ha="right",
            va="bottom",
            zorder=20,
            bbox=dict(
                facecolor="black",
                edgecolor="#b89b00",
                linewidth=0.5,
                boxstyle="square,pad=0.2",
                alpha=0.85,
            ),
        )


def satellite_tick():
    """Timer callback: pick up newly arrived tiles / a changed view."""
    if fig is None or ax is None:
        return

    if not state["sat_dirty"]:
        return

    # If a full redraw is in progress, try again on the next tick.
    if not draw_lock.acquire(blocking=False):
        return

    try:
        update_satellite()
        fig.canvas.draw_idle()

    except Exception as exc:
        log_status(f"SATELLITE ERROR -> {type(exc).__name__}: {exc}")

    finally:
        draw_lock.release()


# ---------------------------------------------------------------------------
# CUSTOM RADAR PALETTE LOADING
# ---------------------------------------------------------------------------

def normalize_custom_palette_order(palette):
    """Keep the file's declared colour order and only normalise the positions.

    Custom .pal/.pals files should be displayed exactly as authored. The user can
    explicitly reverse a palette via the REVERSE button if they want the colours
    flipped in the opposite direction.
    """
    normalized = []

    for index, entry in enumerate(palette):
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            pos, value = entry
            if isinstance(pos, (int, float)):
                pos = float(pos)
            else:
                pos = index / (len(palette) - 1) if len(palette) > 1 else 0.0
        else:
            pos = index / (len(palette) - 1) if len(palette) > 1 else 0.0
            value = entry

        if pos < 0.0:
            pos = 0.0
        elif pos > 1.0:
            pos = 1.0

        normalized.append((pos, value))

    return normalized


def parse_pals_palette(path):
    """Read a .pals/.pal file and convert it to Matplotlib colour entries."""
    palette = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            lower = line.lower()
            if lower.startswith("#") or lower.startswith(";"):
                continue

            if lower.startswith(("product:", "units:", "step:")):
                continue
            if lower.startswith(("jbr", "palette", "version")):
                continue

            prefix = ""
            rest = line

            if ":" in line:
                prefix, rest = line.split(":", 1)
                prefix = prefix.strip().lower()
                rest = rest.strip()
            else:
                prefix = lower.strip()

            if prefix and not prefix.startswith(("solidcolor", "color")):
                if not line.replace(" ", "").replace("\t", "").isdigit():
                    continue

            numbers = [
                float(v)
                for v in re.findall(r"[-+]?(?:\d+\.\d+|\d+)", rest if rest else line)
            ]

            if len(numbers) < 3:
                continue

            rgb = None
            if len(numbers) >= 4 and all(0 <= channel <= 255 for channel in numbers[1:4]):
                rgb = tuple(channel / 255.0 for channel in numbers[1:4])
            elif len(numbers) >= 3 and all(0 <= channel <= 255 for channel in numbers[:3]):
                rgb = tuple(channel / 255.0 for channel in numbers[:3])

            if rgb is None:
                continue

            palette.append(rgb)

    if not palette:
        raise ValueError(
            "No usable RGB entries found in the palette file. "
            "Expected a standard .pals or .pal colour table."
        )

    normalized = []
    for index, rgb in enumerate(palette):
        r, g, b = rgb
        pos = 0.0 if len(palette) == 1 else index / (len(palette) - 1)

        normalized.append((pos, "#{:02x}{:02x}{:02x}".format(
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)),
        )))

    return normalized


def apply_custom_palette(palette, name=None):
    """Store a palette and immediately refresh the radar legend and image."""
    state["custom_radar_palette"] = normalize_custom_palette_order(palette)

    if name is not None:
        state["custom_radar_name"] = name

    if state["product"] == "radar":
        if state.get("cax") is not None:
            draw_radar_colourbar()
        redraw()
        if fig is not None:
            fig.canvas.draw_idle()


def load_custom_palette(event=None):
    """Prompt the user to select a radar palette (.pals or .pal) file."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select radar palette (.pals or .pal)",
            filetypes=[
                ("Radar palette files", "*.pals"),
                ("Radar palette files", "*.pal"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()

        if not path:
            return

        palette = parse_pals_palette(path)
        apply_custom_palette(palette, os.path.basename(path))

        log_status(
            f"PALETTE -> Loaded custom radar table '{state['custom_radar_name']}'"
        )

    except Exception as exc:
        log_status(f"PALETTE ERROR -> {type(exc).__name__}: {exc}")


def reverse_custom_palette(event=None):
    """Flip the active custom palette so warm colours can be moved to the high end."""
    if not state.get("custom_radar_palette"):
        return

    palette = list(reversed(state["custom_radar_palette"]))
    apply_custom_palette(palette)
    log_status("PALETTE -> Reversed custom radar table direction.")


# ---------------------------------------------------------------------------
# RADAR COLOUR MAPS
# ---------------------------------------------------------------------------

def get_radar_norm():
    return colors.PowerNorm(
        gamma=0.85,
        vmin=RAIN_VMIN,
        vmax=RAIN_VMAX,
        clip=True,
    )


def set_no_rain_colour(cmap):
    """Colour used where there is no rain.

    It must be fully transparent in normal mode, otherwise the radar image
    paints an opaque black sheet over the whole UK and hides the satellite
    imagery underneath. FAX mode has no imagery, so it stays solid white.
    """
    if state["fax_mode"]:
        cmap.set_bad("white", 1.0)
    else:
        cmap.set_bad((0.0, 0.0, 0.0, 0.0))


def get_radar_colormap():

    if state.get("custom_radar_palette"):
        cmap = colors.LinearSegmentedColormap.from_list(
            "custom_radar",
            state["custom_radar_palette"],
            N=256,
        )
        set_no_rain_colour(cmap)
        return cmap

    if state["fax_mode"]:

        palette = [
            (0.00, "#757575"),
            (0.08, "#757575"),
            (0.16, "#666666"),
            (0.26, "#666666"),
            (0.36, "#424242"),
            (0.48, "#424242"),
            (0.58, "#212121"),
            (0.68, "#212121"),
            (0.76, "#1a1a1a"),
            (0.84, "#1a1a1a"),
            (0.90, "#121212"),
            (0.96, "#121212"),
            (1.00, "#000000"),
        ]

        cmap = colors.LinearSegmentedColormap.from_list(
            "fax_radar", palette, N=256
        )

    else:

        palette = [
            (0.00, "#00bbe1"),
            (0.06, "#0099c8"),
            (0.12, "#004d6e"),
            (0.19, "#00205f"),
            (0.25, "#ffeb00"),
            (0.31, "#fab900"),
            (0.38, "#ff8c28"),
            (0.44, "#f56400"),
            (0.50, "#e60000"),
            (0.56, "#b40000"),
            (0.62, "#820000"),
            (0.69, "#500000"),
            (0.75, "#bf2a7f"),
            (0.81, "#ff55ff"),
            (0.88, "#ff7aff"),
            (0.94, "#ffb3ff"),
            (1.00, "#ffd9ff"),
        ]

        cmap = colors.LinearSegmentedColormap.from_list(
            "retro_radar", palette, N=256
        )

    set_no_rain_colour(cmap)

    return cmap


# ---------------------------------------------------------------------------
# FAX RADAR LEVELS
# ---------------------------------------------------------------------------

def get_fax_radar_levels():
    return [0.05, 0.20, 0.50, 1.00, 2.00, 4.00, 8.00, 16.00]


def get_fax_radar_colours():
    return [
        "#ffffff",
        "#eeeeee",
        "#bdbdbd",
        "#666666",
        "#1f4e79",
        "#003b73",
        "#000000",
    ]


# ---------------------------------------------------------------------------
# RADAR LEGEND
# ---------------------------------------------------------------------------

def hide_radar_colourbar():
    if state["cbar"] is not None:
        try:
            state["cbar"].remove()
        except Exception:
            pass
        state["cbar"] = None

    if state["cax"] is not None:
        state["cax"].set_visible(False)


def draw_radar_colourbar():
    if state["cax"] is None:
        return

    if state["cbar"] is not None:
        try:
            state["cbar"].remove()
        except Exception:
            pass
        state["cbar"] = None

    if not state["fax_mode"]:
        cmap = get_radar_colormap()
        norm = get_radar_norm()

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        state["cbar"] = fig.colorbar(
            sm,
            cax=state["cax"],
            orientation="horizontal",
            ticks=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
        )

        state["cbar"].set_label(
            "RAINFALL RATE (mm/h)",
            color="white",
            fontsize=8,
            fontname=RETRO_FONT,
        )

        state["cbar"].ax.tick_params(
            colors="white",
            labelsize=7,
            length=3,
            width=0.8,
        )
        state["cbar"].ax.set_facecolor("black")
        state["cbar"].outline.set_edgecolor("#b89b00")
        state["cbar"].outline.set_linewidth(0.8)
        state["cax"].set_visible(True)
        return

    boundaries = get_fax_radar_levels()
    colours = get_fax_radar_colours()

    cmap = colors.ListedColormap(colours)
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    state["cbar"] = fig.colorbar(
        sm,
        cax=state["cax"],
        orientation="horizontal",
        boundaries=boundaries,
        ticks=boundaries,
    )

    state["cbar"].set_label(
        "RAINFALL RATE (mm/h)",
        color="black",
        fontsize=8,
        fontname=RETRO_FONT,
    )

    state["cbar"].ax.tick_params(
        colors="black",
        labelsize=7,
        length=3,
        width=0.8,
    )

    state["cbar"].ax.set_facecolor("white")

    state["cax"].set_visible(True)


# ---------------------------------------------------------------------------
# RADAR LAYER
# ---------------------------------------------------------------------------

def get_radar_regrid_shape():
    """Radar image size (shorter side, in pixels) to match the current view."""
    pixels = int(min(ax.bbox.width, ax.bbox.height) * RADAR_REGRID_SCALE)
    return max(RADAR_REGRID_MIN, min(RADAR_REGRID_MAX, pixels))


def add_radar_layer(alpha=1.0, zorder=0):
    """Draw the radar rain image for the CURRENT view and return the artist."""
    (
        data,
        projdef,
        ul_lat,
        ul_lon,
        lr_lat,
        lr_lon,
        date,
        starttime
    ) = state["payload"]

    radar_pyproj = CRS.from_proj4(projdef)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        radar_pyproj,
        always_xy=True
    )

    ul_x, ul_y = transformer.transform(ul_lon, ul_lat)
    lr_x, lr_y = transformer.transform(lr_lon, lr_lat)

    radar_crs = ccrs.TransverseMercator(
        central_longitude=-2,
        central_latitude=49,
        scale_factor=0.999601,
        false_easting=400000,
        false_northing=-100000,
        globe=ccrs.Globe(ellipse="airy")
    )

    rain = np.ma.masked_invalid(data)
    rain = np.ma.masked_less(rain, RAIN_VMIN)

    artist = ax.imshow(
        rain,
        extent=[ul_x, lr_x, lr_y, ul_y],
        origin="upper",
        transform=radar_crs,
        cmap=get_radar_colormap(),
        norm=get_radar_norm(),
        interpolation="nearest",
        alpha=alpha,
        zorder=zorder,
        regrid_shape=get_radar_regrid_shape()
    )

    state["radar_artist"] = artist
    state["radar_view"] = get_map_extent()
    state["radar_style"] = (alpha, zorder)

    return artist


def view_changed(a, b):
    if a is None or b is None:
        return True

    tolerance = 0.002 * abs(a[1] - a[0])

    return any(abs(x - y) > tolerance for x, y in zip(a, b))


def refresh_radar_layer():
    """Re-project the radar for the current view so zoom / pan stay sharp."""
    try:
        with draw_lock:

            old = state["radar_artist"]

            if state["payload"] is None or old is None:
                return

            if state["product"] not in ("radar", "lightning"):
                return

            alpha, zorder = state["radar_style"]

            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()

            # Add the new image before removing the old one: no blank flash.
            add_radar_layer(alpha, zorder)

            try:
                old.remove()
            except Exception:
                pass

            # imshow can change the view; put it back exactly as it was.
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            fig.canvas.draw_idle()

    except Exception as exc:
        log_status(f"RENDER ERROR -> {type(exc).__name__}: {exc}")

    finally:
        state["radar_busy"] = False


def radar_view_tick():
    """Timer callback: re-draw the radar once the view has settled."""
    if fig is None or ax is None or executor is None:
        return

    if state["radar_artist"] is None or state["payload"] is None:
        return

    if state["radar_busy"] or state["is_panning"]:
        return

    if time.time() - state["view_changed_at"] < RADAR_SETTLE_SECONDS:
        return

    if not view_changed(get_map_extent(), state["radar_view"]):
        return

    state["radar_busy"] = True

    executor.submit(refresh_radar_layer)


# ---------------------------------------------------------------------------
# RADAR RENDERING
# ---------------------------------------------------------------------------

def draw_radar():

    log_status("RENDER -> Drawing UK Radar Rain Rate overlay...")

    current_extent = get_map_extent()

    ax.clear()

    date, starttime = state["payload"][6:8]

    ax.set_facecolor("white" if state["fax_mode"] else "black")

    set_map_extent(current_extent)

    draw_basemap()

    add_radar_layer(alpha=1.0, zorder=0)

    if state["fax_mode"]:
        coastline_colour = "#202020"
        grid_colour = "#aaaaaa"
    else:
        coastline_colour = "#b89b00"
        grid_colour = "white"

    ax.coastlines(
        resolution="10m",
        color=coastline_colour,
        linewidth=0.6
    )

    ax.gridlines(
        draw_labels=False,
        linewidth=0.25,
        color=grid_colour,
        alpha=(0.45 if state["fax_mode"] else 0.2)
    )

    timestamp = (
        f"{date[:4]}-{date[4:6]}-{date[6:8]} "
        f"{starttime[:2]}:{starttime[2:4]} UTC"
    )

    set_win95_title(f"UK RADAR RAINFALL RATE | {timestamp} |")

    draw_radar_colourbar()

    log_status("IDLE -> RADAR RENDERING COMPLETE")


# ---------------------------------------------------------------------------
# REAL-TIME LIGHTNING
# ---------------------------------------------------------------------------

def _lon_to_tile_x(lon, zoom):
    n = 2 ** zoom
    return (lon + 180.0) / 360.0 * n


def _lat_to_tile_y(lat, zoom):
    lat = max(-85.05112878, min(85.05112878, lat))
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n


def _tile_bounds_web_mercator(x, y, zoom):
    n = 2 ** zoom
    world = 20037508.342789244 * 2.0

    x0 = (x / n) * world - 20037508.342789244
    x1 = ((x + 1) / n) * world - 20037508.342789244

    y1 = 20037508.342789244 - (y / n) * world
    y0 = 20037508.342789244 - ((y + 1) / n) * world

    return x0, x1, y0, y1


def fetch_lightning_image():
    """Fetch the live LightningMaps UK image and return it with a map extent."""
    request = Request(
        LIGHTNING_URL.format(int(time.time())),
        headers={"User-Agent": "UK-Retro-Radar/0.2.0"},
    )

    with urlopen(request, timeout=20) as response:
        image_bytes = response.read()
        content_type = response.headers.get("Content-Type", "")

    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        preview = image_bytes[:80].decode("utf-8", errors="replace").replace(
            "\n", " "
        )
        raise RuntimeError(
            "Lightning image server did not return a PNG "
            f"(Content-Type: {content_type}). Response starts with: {preview!r}"
        )

    image = mpimg.imread(BytesIO(image_bytes))
    return np.asarray(image), list(LIGHTNING_EXTENT)


def draw_lightning():
    """Render the live lightning page in the main map area."""
    log_status("RENDER -> Drawing real-time UK lightning map...")

    current_extent = get_map_extent()

    ax.clear()

    if state["fax_mode"]:
        ax.set_facecolor("white")
        map_text_colour = "black"
        edge_colour = "#404040"
    else:
        ax.set_facecolor("black")
        map_text_colour = "#00ffff"
        edge_colour = "#b89b00"

    set_map_extent(current_extent)

    draw_basemap()
    hide_radar_colourbar()

    if state.get("payload") is not None:
        add_radar_layer(alpha=0.18, zorder=1)

    lightning_data = state.get("lightning_image")

    if lightning_data is None:
        ax.text(
            0.5,
            0.5,
            "LIGHTNING FEED\nWAITING FOR DATA...",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=map_text_colour,
            fontsize=16,
            fontname=RETRO_FONT,
        )
    else:
        image, image_extent = lightning_data

        ax.imshow(
            image,
            extent=image_extent,
            origin="upper",
            interpolation="nearest",
            transform=ccrs.PlateCarree(),
            zorder=5,
            alpha=0.9,
        )

    ax.coastlines(
        resolution="10m",
        color=edge_colour,
        linewidth=0.6,
    )

    ax.gridlines(
        draw_labels=False,
        linewidth=0.25,
        color=edge_colour,
        alpha=0.25,
    )

    # Shorter title so it fits a narrow (portrait) window.
    set_win95_title("UK LIGHTNING | LIVE | BLITZORTUNG")

    ax.text(
        0.01,
        0.015,
        "LIGHTNING DATA: LIGHTNINGMAPS / BLITZORTUNG",
        transform=ax.transAxes,
        color=map_text_colour,
        fontsize=7,
        fontname=RETRO_FONT,
        ha="left",
        va="bottom",
        bbox=dict(
            facecolor="black" if not state["fax_mode"] else "white",
            edgecolor=edge_colour,
            linewidth=0.5,
            boxstyle="square,pad=0.2",
            alpha=0.85,
        ),
    )

    log_status("IDLE -> LIGHTNING RENDERING COMPLETE")


def draw_warnings():
    """Render the warnings log as a dedicated page."""
    ax.clear()

    if state["fax_mode"]:
        ax.set_facecolor("white")
        text_color = "black"
        panel_color = "#f2f2f2"
        accent = "#404040"
    else:
        ax.set_facecolor("black")
        text_color = "#f5f5f5"
        panel_color = "#111111"
        accent = "#b89b00"

    set_map_extent(get_default_extent())

    draw_basemap()
    hide_radar_colourbar()

    if state["warnings"]:
        # Wrap long lines so they stay inside a narrow portrait panel.
        import textwrap
        wrapped = [
            "\n".join(textwrap.wrap(item, width=40)) or item
            for item in state["warnings"]
        ]
        body = "\n\n".join(wrapped)
        empty_message = False
    else:
        body = "NO WARNINGS\n\n"
        empty_message = True

    ax.text(
        0.5,
        0.5,
        body,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=text_color,
        fontsize=9 if not empty_message else 14,
        fontname=RETRO_FONT,
        linespacing=1.4,
        bbox=dict(
            facecolor=panel_color,
            edgecolor=accent,
            linewidth=1.0,
            boxstyle="square,pad=0.5",
            alpha=0.9,
        ),
    )

    ax.coastlines(
        resolution="10m",
        color=accent,
        linewidth=0.5,
    )

    ax.gridlines(
        draw_labels=False,
        linewidth=0.2,
        color=accent,
        alpha=0.25,
    )

    set_win95_title("SYSTEM WARNINGS")


def load_lightning_async():
    """Fetch lightning data without blocking the GUI."""
    try:
        log_status("LIGHTNING -> Fetching current live map...")

        image = fetch_lightning_image()

        state["lightning_image"] = image

        if state["product"] == "lightning":
            redraw()

    except Exception as exc:
        log_status(f"LIGHTNING ERROR -> {type(exc).__name__}: {exc}")

    finally:
        state["is_loading"] = False


def choose_lightning(event=None):
    """Switch the main page to the live lightning display."""
    if state["is_loading"]:
        return

    state["product"] = "lightning"
    state["is_loading"] = True

    log_status("USER EVENT -> SWITCHING TO REAL-TIME LIGHTNING...")

    fig.canvas.draw_idle()

    executor.submit(load_lightning_async)


def choose_radar(event=None):
    """Return to the radar display."""
    if state["is_loading"]:
        return

    choose_product("radar")


def choose_warnings(event=None):
    """Switch to the warnings log page."""
    if state["is_loading"]:
        return

    state["product"] = "warnings"
    redraw()


def refresh_lightning(event=None):
    """Refresh the live lightning image."""
    if state["product"] != "lightning":
        return

    if state["is_loading"]:
        return

    state["is_loading"] = True

    executor.submit(load_lightning_async)


# ---------------------------------------------------------------------------
# REDRAW
# ---------------------------------------------------------------------------

def redraw():

    with draw_lock:

        if state["product"] == "lightning":
            draw_lightning()
        elif state["product"] == "warnings":
            draw_warnings()
        else:
            draw_radar()
            update_fax_button()

        fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# FAX MODE
# ---------------------------------------------------------------------------

def toggle_fax(event=None):

    state["fax_mode"] = not state["fax_mode"]

    if state["fax_mode"]:
        log_status("FAX MODE -> HIGH-CONTRAST WHITE / INK DISPLAY ENABLED.")
    else:
        log_status("FAX MODE -> NORMAL RADAR DISPLAY RESTORED.")

    redraw()


def update_fax_button():

    if state["fax_mode"]:
        fax_button_ax.set_facecolor("#606060")
        fax_button.label.set_color("white")
        fax_button.label.set_text("FAX ON")
    else:
        fax_button_ax.set_facecolor("#d0d0d0")
        fax_button.label.set_color("black")
        fax_button.label.set_text("FAX")


# ---------------------------------------------------------------------------
# PRODUCT SELECTION
# ---------------------------------------------------------------------------

def choose_product(product):

    if product != "radar":
        return

    if state["is_loading"]:
        return

    state["is_loading"] = True

    log_status("USER EVENT -> LOADING RADAR DATA...")

    state["product"] = product

    fig.canvas.draw_idle()

    def async_job():

        try:
            key, filename, payload = download_worker(product)

            if key is None:
                log_status("USER EVENT -> RADAR DATA IS CURRENTLY UNAVAILABLE.")
                return

            state["key"] = key
            state["filename"] = filename
            state["payload"] = payload

            redraw()

        except Exception as exc:
            log_status(f"LOAD ERROR -> {type(exc).__name__}: {exc}")

        finally:
            state["is_loading"] = False

    executor.submit(async_job)


# ---------------------------------------------------------------------------
# AUTOMATIC RADAR UPDATE
# ---------------------------------------------------------------------------

def check_for_new_radar():

    try:
        new_key = find_latest_radar()

        if new_key is None:
            return

        if new_key == state["key"]:
            log_status("NO UPDATE -> RADAR FILE UNCHANGED.")
            return

        log_status("NEW RADAR -> A NEWER RADAR FILE IS AVAILABLE.")

        filename = os.path.basename(new_key)

        log_status(f"DOWNLOAD -> FETCHING {filename} FROM AMAZON S3...")

        s3.download_file(RADAR_BUCKET, new_key, filename)

        payload = load_radar(filename)

        state["key"] = new_key
        state["filename"] = filename
        state["payload"] = payload

        redraw()

    except Exception as exc:
        log_status(f"UPDATE ERROR -> {type(exc).__name__}: {exc}")

    finally:
        state["is_loading"] = False


# ---------------------------------------------------------------------------
# MANUAL / TIMER UPDATE
# ---------------------------------------------------------------------------

def update(event=None):

    if state["is_loading"]:
        return

    if state["product"] == "lightning":
        refresh_lightning()
        return

    if state["product"] != "radar":
        return

    log_status("TIMER EVENT -> CHECKING AWS S3 FOR NEW RADAR DATA...")

    state["is_loading"] = True

    executor.submit(check_for_new_radar)


# ---------------------------------------------------------------------------
# PORTRAIT LAYOUT
# ---------------------------------------------------------------------------

def position_buttons(event=None):
    """Lay the buttons out in a grid (BUTTON_COLS per row) above the status bar."""
    button_axes = [
        update_button_ax,
        reset_button_ax,
        fax_button_ax,
        radar_button_ax,
        palette_button_ax,
        reverse_palette_button_ax,
        lightning_button_ax,
        warning_button_ax,
    ]

    if any(b is None for b in button_axes):
        return

    button_w = (
        (1.0 - 2 * BUTTON_X - (BUTTON_COLS - 1) * BUTTON_GAP)
        / BUTTON_COLS
    )

    rows = math.ceil(len(button_axes) / BUTTON_COLS)

    bottom_y = STATUS_BAR_Y + STATUS_BAR_H + BUTTON_ROW_GAP

    for index, button_ax in enumerate(button_axes):
        row = index // BUTTON_COLS
        col = index % BUTTON_COLS

        # Row 0 is the top row of the grid.
        y = bottom_y + (rows - 1 - row) * (BUTTON_H + BUTTON_ROW_GAP)
        x = BUTTON_X + col * (button_w + BUTTON_GAP)

        button_ax.set_position([x, y, button_w, BUTTON_H])


def position_layout(event=None):
    """Make the map fill the whole area above the controls (no black bars).

    The axes always take the full available rectangle. If the window shape
    changes, the visible extent is adjusted (keeping the same centre and the
    same north-south span) so the map still fills that rectangle exactly.
    """
    if fig is None or ax is None:
        return

    left = 0.005
    right = 0.995
    bottom = MAP_BOTTOM
    top = MAP_TOP

    ax.set_position([left, bottom, right - left, top - bottom])

    fig_width, fig_height = fig.get_size_inches()
    box_aspect = (
        (right - left) * fig_width
        / ((top - bottom) * fig_height)
    )

    try:
        x_min, x_max, y_min, y_max = get_map_extent()

        centre_x = (x_min + x_max) / 2
        height = abs(y_max - y_min)

        if height > 0:
            width = height * box_aspect
            set_map_extent([
                centre_x - width / 2,
                centre_x + width / 2,
                y_min,
                y_max,
            ])
    except Exception:
        pass

    mark_view_changed()

    position_buttons(event)

    fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------------

def main():

    global fig
    global ax
    global update_button_ax
    global reset_button_ax
    global fax_button_ax
    global radar_button_ax
    global palette_button_ax
    global reverse_palette_button_ax
    global lightning_button_ax
    global warning_button_ax
    global update_button
    global reset_button
    global fax_button
    global radar_button
    global palette_button
    global reverse_palette_button
    global lightning_button
    global warning_button
    global status_bar_ax
    global auto_timer
    global lightning_timer
    global sat_timer
    global s3
    global executor
    global sat_executor

    # -----------------------------------------------------------------------
    # AWS S3
    # -----------------------------------------------------------------------

    s3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        region_name="eu-west-2"
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    # Separate pool for satellite tiles so they never hold up radar downloads.
    sat_executor = concurrent.futures.ThreadPoolExecutor(max_workers=SAT_WORKERS)

    # -----------------------------------------------------------------------
    # FIGURE (portrait)
    # -----------------------------------------------------------------------

    fig = plt.figure(
        figsize=FIGURE_SIZE,
        facecolor="black"
    )

    # -----------------------------------------------------------------------
    # CARTOPY MAP
    # -----------------------------------------------------------------------

    ax = fig.add_axes(
        [0.01, MAP_BOTTOM, 0.98, 0.77],
        projection=MAP_CRS,
        facecolor="black"
    )

    ax.set_aspect("equal", adjustable="box")

    # -----------------------------------------------------------------------
    # COLOUR BAR (shown beneath the radar map in both normal and FAX modes)
    # -----------------------------------------------------------------------

    state["cax"] = fig.add_axes(
        [0.10, CBAR_Y, 0.80, CBAR_H]
    )

    state["cax"].set_visible(False)

    state["cbar"] = None

    # -----------------------------------------------------------------------
    # STATUS BAR
    # -----------------------------------------------------------------------

    status_bar_ax = fig.add_axes(
        [0.01, STATUS_BAR_Y, 0.98, STATUS_BAR_H]
    )

    status_bar_ax.set_facecolor("#c0c0c0")

    status_bar_ax.set_xticks([])
    status_bar_ax.set_yticks([])

    status_bar_ax.spines["top"].set_color("#808080")
    status_bar_ax.spines["left"].set_color("#808080")
    status_bar_ax.spines["bottom"].set_color("#ffffff")
    status_bar_ax.spines["right"].set_color("#ffffff")

    status_bar_ax.spines["top"].set_linewidth(2)
    status_bar_ax.spines["left"].set_linewidth(2)

    state["status_text"] = status_bar_ax.text(
        0.012,
        0.5,
        "SYSTEM INITIALIZING...",
        color="black",
        fontsize=7,
        fontname=RETRO_FONT,
        fontweight="normal",
        verticalalignment="center",
        linespacing=1.3
    )

    # -----------------------------------------------------------------------
    # BUTTONS (positions are set by position_buttons())
    # -----------------------------------------------------------------------

    def make_button(label, callback, fontsize):
        button_ax = fig.add_axes([0.0, 0.0, 0.1, BUTTON_H])
        button = Button(
            button_ax,
            label,
            color="#d0d0d0",
            hovercolor="#e0e0e0"
        )
        style_button(button_ax, button, fontsize=fontsize)
        button.on_clicked(callback)
        return button_ax, button

    update_button_ax, update_button = make_button("UPDATE", update, 8)
    reset_button_ax, reset_button = make_button("RESET VIEW", reset_view, 8)
    fax_button_ax, fax_button = make_button("FAX", toggle_fax, 8)
    radar_button_ax, radar_button = make_button("RADAR", choose_radar, 8)
    palette_button_ax, palette_button = make_button(
        "PALETTE", load_custom_palette, 8
    )
    reverse_palette_button_ax, reverse_palette_button = make_button(
        "REVERSE", reverse_custom_palette, 8
    )
    lightning_button_ax, lightning_button = make_button(
        "LIGHTNING", choose_lightning, 8
    )
    warning_button_ax, warning_button = make_button(
        "WARNINGS", choose_warnings, 8
    )

    # -----------------------------------------------------------------------
    # RESIZE HANDLER
    # -----------------------------------------------------------------------

    fig.canvas.mpl_connect("resize_event", position_layout)

    # Note: the window is deliberately NOT maximised any more, so it opens
    # as a tall portrait window. Drag it to any size and the layout adapts.

    position_layout()

    # -----------------------------------------------------------------------
    # MOUSE CONTROLS
    # -----------------------------------------------------------------------

    fig.canvas.mpl_connect("scroll_event", zoom_map)
    fig.canvas.mpl_connect("button_press_event", pan_start)
    fig.canvas.mpl_connect("motion_notify_event", pan_move)
    fig.canvas.mpl_connect("button_release_event", pan_end)

    # -----------------------------------------------------------------------
    # INITIAL VIEW
    # -----------------------------------------------------------------------

    set_default_view()

    # -----------------------------------------------------------------------
    # START RADAR
    # -----------------------------------------------------------------------

    choose_product("radar")

    # -----------------------------------------------------------------------
    # AUTOMATIC UPDATE TIMER
    # -----------------------------------------------------------------------

    auto_timer = fig.canvas.new_timer(
        interval=AUTO_UPDATE_MINUTES * 60 * 1000
    )

    auto_timer.add_callback(update)
    auto_timer.start()

    # -----------------------------------------------------------------------
    # REAL-TIME LIGHTNING TIMER
    # -----------------------------------------------------------------------

    lightning_timer = fig.canvas.new_timer(
        interval=LIGHTNING_REFRESH_SECONDS * 1000
    )

    lightning_timer.add_callback(refresh_lightning)
    lightning_timer.start()

    # -----------------------------------------------------------------------
    # SATELLITE IMAGERY TIMER
    # -----------------------------------------------------------------------

    sat_timer = fig.canvas.new_timer(
        interval=SAT_REFRESH_MS
    )

    sat_timer.add_callback(satellite_tick)
    sat_timer.add_callback(radar_view_tick)
    sat_timer.start()

    # -----------------------------------------------------------------------
    # RUN GUI
    # -----------------------------------------------------------------------

    plt.show()


# ---------------------------------------------------------------------------
# START PROGRAM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
