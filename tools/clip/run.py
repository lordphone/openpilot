#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess
import threading
import queue
import multiprocessing
import itertools
import numpy as np
import tqdm
from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from openpilot.tools.lib.route import Route
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.filereader import FileReader
from openpilot.tools.lib.framereader import FrameReader, ffprobe
from openpilot.selfdrive.test.process_replay.migration import migrate_all
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.utils import Timer
from msgq.visionipc import VisionIpcServer, VisionStreamType

FRAMERATE = 20
DEMO_ROUTE, DEMO_START, DEMO_END = 'a2a0ccea32023010/2023-07-27--13-01-19', 90, 105

logger = logging.getLogger('clip')


def parse_args():
  parser = ArgumentParser(description="Direct clip renderer")
  parser.add_argument("route", nargs="?", help="Route ID (dongle/route or dongle/route/start/end)")
  parser.add_argument("-s", "--start", type=int, help="Start time in seconds")
  parser.add_argument("-e", "--end", type=int, help="End time in seconds")
  parser.add_argument("-o", "--output", default="output.mp4", help="Output file path")
  parser.add_argument("-d", "--data-dir", help="Local directory with route data")
  parser.add_argument("-t", "--title", help="Title overlay text")
  parser.add_argument("-f", "--file-size", type=float, default=9.0, help="Target file size in MB")
  parser.add_argument("-x", "--speed", type=int, default=1, help="Speed multiplier")
  parser.add_argument("--demo", action="store_true", help="Use demo route with default timing")
  parser.add_argument("--big", action="store_true", default=True, help="Use big UI (2160x1080)")
  parser.add_argument("--qcam", action="store_true", help="Use qcamera instead of fcamera")
  parser.add_argument("--windowed", action="store_true", help="Show window")
  parser.add_argument("--no-metadata", action="store_true", help="Disable metadata overlay")
  parser.add_argument("--no-time-overlay", action="store_true", help="Disable time overlay")
  parser.add_argument("--parallel", action="store_true", help="Use parallel GOP decoding (faster but uses more memory)")
  parser.add_argument("--workers", type=int, default=None, help="Number of parallel decode workers (default: auto)")
  args = parser.parse_args()

  if args.demo:
    args.route, args.start, args.end = args.route or DEMO_ROUTE, args.start or DEMO_START, args.end or DEMO_END
  elif not args.route:
    parser.error("route is required (or use --demo)")

  if args.route and args.route.count('/') == 3:
    parts = args.route.split('/')
    args.route, args.start, args.end = '/'.join(parts[:2]), args.start or int(parts[2]), args.end or int(parts[3])

  if args.start is None or args.end is None:
    parser.error("--start and --end are required")
  if args.end <= args.start:
    parser.error(f"end ({args.end}) must be greater than start ({args.start})")
  return args


def setup_env(output_path: str, big: bool = False, speed: int = 1, target_mb: float = 0, duration: int = 0):
  os.environ.update({"RECORD": "1", "OFFSCREEN": "1", "RECORD_OUTPUT": str(Path(output_path).with_suffix(".mp4"))})
  if speed > 1:
    os.environ["RECORD_SPEED"] = str(speed)
  if target_mb > 0 and duration > 0:
    os.environ["RECORD_BITRATE"] = f"{int(target_mb * 8 * 1024 / (duration / speed))}k"
  if big:
    os.environ["BIG"] = "1"


def _download_segment(path: str) -> bytes:
  from openpilot.tools.lib.filereader import FileReader
  with FileReader(path) as f:
    return bytes(f.read())


def _parse_and_chunk_segment(args: tuple) -> list[dict]:
  raw_data, fps = args[0], args[2]
  from openpilot.tools.lib.logreader import _LogFileReader
  messages = migrate_all(list(_LogFileReader("", dat=raw_data, sort_by_time=True)))
  if not messages:
    return []

  dt_ns, chunks, current, next_time = 1e9 / fps, [], {}, messages[0].logMonoTime + 1e9 / fps  # type: ignore[var-annotated]
  for msg in messages:
    if msg.logMonoTime >= next_time:
      chunks.append(current)
      current, next_time = {}, next_time + dt_ns * ((msg.logMonoTime - next_time) // dt_ns + 1)
    current[msg.which()] = msg
  return chunks + [current] if current else chunks


def load_logs_parallel(log_paths: list[str], fps: int = 20) -> list[dict]:
  num_workers = min(16, len(log_paths), (multiprocessing.cpu_count() or 1))
  logger.info(f"Downloading {len(log_paths)} segments with {num_workers} workers...")

  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    futures = {pool.submit(_download_segment, path): idx for idx, path in enumerate(log_paths)}
    raw_data = {futures[f]: f.result() for f in as_completed(futures)}

  logger.info("Parsing and chunking segments...")
  with multiprocessing.Pool(num_workers) as pool:
    return list(itertools.chain.from_iterable(pool.map(_parse_and_chunk_segment, [(raw_data[i], i, fps) for i in range(len(log_paths))])))


def patch_submaster(message_chunks, ui_state):
  # Reset started_frame so alerts render correctly (recv_frame must be >= started_frame)
  ui_state.started_frame = 0
  ui_state.started_time = time.monotonic()

  def mock_update(timeout=None):
    sm, t = ui_state.sm, time.monotonic()
    sm.updated = dict.fromkeys(sm.services, False)
    if sm.frame < len(message_chunks):
      for svc, msg in message_chunks[sm.frame].items():
        if svc in sm.data:
          sm.seen[svc] = sm.updated[svc] = sm.alive[svc] = sm.valid[svc] = True
          sm.data[svc] = getattr(msg.as_builder(), svc)
          sm.logMonoTime[svc], sm.recv_time[svc], sm.recv_frame[svc] = msg.logMonoTime, t, sm.frame
    sm.frame += 1
  ui_state.sm.update = mock_update


def get_frame_dimensions(camera_path: str) -> tuple[int, int]:
  """Get frame dimensions from a video file using ffprobe."""
  probe = ffprobe(camera_path)
  stream = probe["streams"][0]
  return stream["width"], stream["height"]


def iter_segment_frames(camera_paths, start_time, end_time, fps=20, use_qcam=False, frame_size: tuple[int, int] | None = None):
  frames_per_seg = fps * 60
  start_frame, end_frame = int(start_time * fps), int(end_time * fps)
  current_seg: int = -1
  seg_frames: FrameReader | np.ndarray | None = None

  for global_idx in range(start_frame, end_frame):
    seg_idx, local_idx = global_idx // frames_per_seg, global_idx % frames_per_seg

    if seg_idx != current_seg:
      current_seg = seg_idx
      path = camera_paths[seg_idx] if seg_idx < len(camera_paths) else None
      if not path:
        raise RuntimeError(f"No camera file for segment {seg_idx}")

      if use_qcam:
        w, h = frame_size or get_frame_dimensions(path)
        with FileReader(path) as f:
          result = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "-", "-f", "rawvideo", "-pix_fmt", "nv12", "-"],
                                  input=f.read(), capture_output=True)
        if result.returncode != 0:
          raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
        seg_frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, w * h * 3 // 2)
      else:
        seg_frames = FrameReader(path, pix_fmt="nv12")

    assert seg_frames is not None
    frame = seg_frames[local_idx] if use_qcam else seg_frames.get(local_idx)  # type: ignore[index, union-attr]
    yield global_idx, frame


class FrameQueue:
  def __init__(self, camera_paths, start_time, end_time, fps=20, prefetch_count=60, use_qcam=False):
    # Probe first valid camera file for dimensions
    first_path = next((p for p in camera_paths if p), None)
    if not first_path:
      raise RuntimeError("No valid camera paths")
    self.frame_w, self.frame_h = get_frame_dimensions(first_path)

    self._queue, self._stop, self._error = queue.Queue(maxsize=prefetch_count), threading.Event(), None
    self._thread = threading.Thread(target=self._worker,
                                    args=(camera_paths, start_time, end_time, fps, use_qcam, (self.frame_w, self.frame_h)), daemon=True)
    self._thread.start()

  def _worker(self, camera_paths, start_time, end_time, fps, use_qcam, frame_size):
    try:
      for idx, data in iter_segment_frames(camera_paths, start_time, end_time, fps, use_qcam, frame_size):
        if self._stop.is_set():
          break
        self._queue.put((idx, data.tobytes()))
    except Exception as e:
      logger.exception("Decode error")
      self._error = e
    finally:
      self._queue.put(None)

  def get(self, timeout=60.0):
    if self._error:
      raise self._error
    result = self._queue.get(timeout=timeout)
    if result is None:
      raise StopIteration("No more frames")
    return result

  def stop(self):
    self._stop.set()
    while not self._queue.empty():
      try:
        self._queue.get_nowait()
      except queue.Empty:
        break
    self._thread.join(timeout=2.0)


def _decode_gop_worker(args):
  """Worker function to decode a single GOP."""
  gop_idx, path, prefix, gop_start, gop_end, gop_global_start, emit_start, emit_end, w, h, pix_fmt = args
  from openpilot.tools.lib.framereader import decompress_video_data
  with FileReader(path) as f:
    f.seek(gop_start)
    raw = prefix + f.read(gop_end - gop_start)
  frames = decompress_video_data(raw, w, h, pix_fmt)
  result = []
  for i, frame in enumerate(frames):
    global_frame_idx = gop_global_start + i
    if global_frame_idx >= emit_end:
      break
    if global_frame_idx < emit_start:
      continue
    result.append((global_frame_idx, frame.tobytes()))
  return gop_idx, result


class ParallelFrameQueue:
  """Frame queue that decodes multiple GOPs in parallel while streaming."""

  def __init__(self, camera_paths, start_time, end_time, fps=20, num_workers=None, use_qcam=False, prefetch_count=60):
    from openpilot.tools.lib.framereader import get_video_index, HEVC_SLICE_I

    self.num_workers = num_workers or min(8, (multiprocessing.cpu_count() or 4))
    self.pix_fmt = "nv12"
    self._queue, self._stop, self._error = queue.Queue(maxsize=prefetch_count), threading.Event(), None

    # Get first valid path and dimensions
    seg_start = int(start_time) // 60
    first_path = camera_paths[seg_start] if seg_start < len(camera_paths) else None
    if not first_path:
      raise RuntimeError("No valid camera paths")

    probe = ffprobe(first_path)
    stream = probe["streams"][0]
    self.frame_w, self.frame_h = stream["width"], stream["height"]

    # Calculate frame ranges
    frames_per_seg = fps * 60
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Build list of GOPs to decode (streaming)
    self._gops = []  # List of (gop_idx, path, prefix, gop_start_byte, gop_end_byte, gop_global_start, emit_start, emit_end, w, h, pix_fmt)
    seg_start_idx = start_frame // frames_per_seg
    seg_end_idx = (end_frame - 1) // frames_per_seg

    logger.info(f"Indexing {seg_end_idx - seg_start_idx + 1} camera segment(s) for parallel decode...")

    gop_idx = 0
    for seg_idx in range(seg_start_idx, seg_end_idx + 1):
      path = camera_paths[seg_idx] if seg_idx < len(camera_paths) else None
      if not path:
        raise RuntimeError(f"No camera file for segment {seg_idx}")
      index_data = get_video_index(path)
      index = index_data["index"]
      prefix = index_data["global_prefix"]
      frame_count = len(index) - 1  # sentinel row at the end
      iframes = np.where(index[:, 0] == HEVC_SLICE_I)[0]
      if len(iframes) == 0:
        raise RuntimeError(f"No I-frames found in segment {seg_idx}")

      local_start = start_frame - seg_idx * frames_per_seg if seg_idx == seg_start_idx else 0
      local_end = end_frame - seg_idx * frames_per_seg if seg_idx == seg_end_idx else frame_count
      local_start = max(0, local_start)
      local_end = min(frame_count, local_end)
      if local_start >= local_end:
        continue

      start_i = np.searchsorted(iframes, local_start, side="right") - 1
      if start_i < 0:
        start_i = 0

      for i in range(start_i, len(iframes)):
        gop_start_frame = int(iframes[i])
        gop_end_frame = frame_count if i + 1 >= len(iframes) else int(iframes[i + 1])
        if gop_end_frame <= local_start:
          continue
        if gop_start_frame >= local_end:
          break

        gop_start_byte = int(index[gop_start_frame, 1])
        gop_end_byte = int(index[gop_end_frame, 1])
        gop_global_start = seg_idx * frames_per_seg + gop_start_frame
        gop_global_end = seg_idx * frames_per_seg + gop_end_frame
        emit_start = max(gop_global_start, start_frame)
        emit_end = min(gop_global_end, end_frame)

        self._gops.append((
          gop_idx,
          path,
          prefix,
          gop_start_byte,
          gop_end_byte,
          gop_global_start,
          emit_start,
          emit_end,
          self.frame_w,
          self.frame_h,
          self.pix_fmt,
        ))
        gop_idx += 1

    if not self._gops:
      raise RuntimeError("No GOPs found for requested clip range")

    logger.info(f"Decoding {len(self._gops)} GOPs in parallel (streaming)...")

    self._thread = threading.Thread(target=self._worker, daemon=True)
    self._thread.start()

  def _worker(self):
    try:
      max_inflight = max(1, min(len(self._gops), self.num_workers * 2))
      pending = {}
      inflight = {}
      next_submit = 0
      next_emit = 0

      def submit_next(pool):
        nonlocal next_submit
        if next_submit >= len(self._gops) or self._stop.is_set():
          return
        fut = pool.submit(_decode_gop_worker, self._gops[next_submit])
        inflight[fut] = next_submit
        next_submit += 1

      with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
        for _ in range(max_inflight):
          submit_next(pool)

        while next_emit < len(self._gops) and not self._stop.is_set():
          if not inflight:
            raise RuntimeError("Parallel decoder stalled with no in-flight tasks")
          done, _ = wait(inflight, timeout=0.1, return_when=FIRST_COMPLETED)
          for fut in done:
            gop_idx, frames = fut.result()
            pending[gop_idx] = frames
            inflight.pop(fut, None)
            submit_next(pool)

          while next_emit in pending and not self._stop.is_set():
            frames = pending.pop(next_emit)
            for item in frames:
              if self._stop.is_set():
                break
              self._queue.put(item)
            next_emit += 1
    except Exception as e:
      logger.exception("Decode error")
      self._error = e
    finally:
      self._queue.put(None)

  def get(self, timeout=60.0):
    if self._error:
      raise self._error
    result = self._queue.get(timeout=timeout)
    if result is None:
      raise StopIteration("No more frames")
    return result

  def stop(self):
    self._stop.set()
    while not self._queue.empty():
      try:
        self._queue.get_nowait()
      except queue.Empty:
        break
    self._thread.join(timeout=2.0)


def load_route_metadata(route):
  from openpilot.common.params import Params, UnknownKeyName
  lr = LogReader(route.log_paths()[0])
  init_data, car_params = lr.first('initData'), lr.first('carParams')

  params = Params()
  for entry in init_data.params.entries:
    try:
      params.put(entry.key, params.cpp2python(entry.key, entry.value))
    except UnknownKeyName:
      pass

  origin = init_data.gitRemote.split('/')[3] if len(init_data.gitRemote.split('/')) > 3 else 'unknown'
  return {
    'version': init_data.version, 'route': route.name.canonical_name,
    'car': car_params.carFingerprint if car_params else 'unknown', 'origin': origin,
    'branch': init_data.gitBranch, 'commit': init_data.gitCommit[:7], 'modified': str(init_data.dirty).lower(),
  }


def draw_text_box(rl, text, x, y, size, gui_app, color=None, center=False):
  box_color, white = rl.Color(0, 0, 0, 85), color or rl.WHITE
  text_width = rl.measure_text(text, size)
  if center:
    x = (gui_app.width - text_width) // 2
  rl.draw_rectangle(x - 8, y - 4, text_width + 16, size + 8, box_color)
  rl.draw_text(text, x, y, size, white)


def render_overlays(rl, gui_app, metadata, title, start_time, frame_idx, show_metadata, show_time):
  if show_metadata and metadata and frame_idx < FRAMERATE * 5:
    m = metadata
    text = ", ".join([f"openpilot v{m['version']}", f"route: {m['route']}", f"car: {m['car']}", f"origin: {m['origin']}",
                      f"branch: {m['branch']}", f"commit: {m['commit']}", f"modified: {m['modified']}"])
    draw_text_box(rl, text, 0, 8, 26, gui_app, center=True)

  if title:
    draw_text_box(rl, title, 0, 60, 48, gui_app, center=True)

  if show_time:
    t = start_time + frame_idx / FRAMERATE
    draw_text_box(rl, f"{int(t)//60:02d}:{int(t)%60:02d}", gui_app.width - rl.measure_text(f"{int(t)//60:02d}:{int(t)%60:02d}", 36) - 45, 45, 36, gui_app)


def clip(route: Route, output: str, start: int, end: int, headless: bool = True, big: bool = False,
         title: str | None = None, show_metadata: bool = True, show_time: bool = True, use_qcam: bool = False,
         parallel: bool = False, num_workers: int | None = None):
  timer, duration = Timer(), end - start

  import pyray as rl
  if big:
    from openpilot.selfdrive.ui.layouts.main import MainLayout
  else:
    from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout as MainLayout  # type: ignore[assignment]
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.system.ui.lib.application import gui_app
  timer.lap("import")

  logger.info(f"Clipping {route.name.canonical_name}, {start}s-{end}s ({duration}s)")
  seg_start, seg_end = start // 60, (end - 1) // 60 + 1
  all_chunks = load_logs_parallel(route.log_paths()[seg_start:seg_end], fps=FRAMERATE)
  timer.lap("logs")

  frame_start = (start - seg_start * 60) * FRAMERATE
  message_chunks = all_chunks[frame_start:frame_start + duration * FRAMERATE]
  if not message_chunks:
    logger.error("No messages to render")
    sys.exit(1)

  metadata = load_route_metadata(route) if show_metadata else None
  if headless:
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)

  with OpenpilotPrefix(shared_download_cache=True):
    camera_paths = route.qcamera_paths() if use_qcam else route.camera_paths()
    if parallel and not use_qcam:
      frame_queue = ParallelFrameQueue(camera_paths, start, end, fps=FRAMERATE, num_workers=num_workers)
    else:
      frame_queue = FrameQueue(camera_paths, start, end, fps=FRAMERATE, use_qcam=use_qcam)
    timer.lap("decode" if parallel else "setup_decode")

    vipc = VisionIpcServer("camerad")
    vipc.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 4, frame_queue.frame_w, frame_queue.frame_h)
    vipc.start_listener()

    patch_submaster(message_chunks, ui_state)
    gui_app.init_window("clip", fps=FRAMERATE)

    main_layout = MainLayout()
    main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    timer.lap("setup")

    frame_idx = 0
    with tqdm.tqdm(total=len(message_chunks), desc="Rendering", unit="frame") as pbar:
      for should_render in gui_app.render():
        if frame_idx >= len(message_chunks):
          break
        _, frame_bytes = frame_queue.get()
        vipc.send(VisionStreamType.VISION_STREAM_ROAD, frame_bytes, frame_idx, int(frame_idx * 5e7), int(frame_idx * 5e7))
        ui_state.update()
        if should_render:
          main_layout.render()
          render_overlays(rl, gui_app, metadata, title, start, frame_idx, show_metadata, show_time)
        frame_idx += 1
        pbar.update(1)
    timer.lap("render")

    frame_queue.stop()
    gui_app.close()
    timer.lap("ffmpeg")

  logger.info(f"Clip saved to: {Path(output).resolve()}")
  logger.info(f"Generated {timer.fmt(duration)}")


def main():
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s\t%(message)s")
  args = parse_args()
  assert args.big, "Clips doesn't support mici UI yet. TODO: make it work"

  setup_env(args.output, big=args.big, speed=args.speed, target_mb=args.file_size, duration=args.end - args.start)
  clip(Route(args.route, data_dir=args.data_dir), args.output, args.start, args.end, not args.windowed,
       args.big, args.title, not args.no_metadata, not args.no_time_overlay, args.qcam,
       args.parallel, args.workers)

if __name__ == "__main__":
  main()
