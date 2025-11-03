#!/usr/bin/env python3

import logging
import os
import platform
import shutil
import sys
import time
import cffi
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
from random import randint
from subprocess import Popen, DEVNULL, PIPE
from typing import Literal

import pyray as rl
import numpy as np

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params, UnknownKeyName
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.logreader import LogReader, ReadMode
from openpilot.tools.lib.route import Route
from msgq.visionipc import VisionIpcServer, VisionStreamType

# GUI imports moved to clip() function after DISPLAY is set

DEFAULT_OUTPUT = 'output.mp4'
DEMO_START = 90
DEMO_END = 105
DEMO_ROUTE = 'a2a0ccea32023010/2023-07-27--13-01-19'
FRAMERATE = 20
RESOLUTION = '2160x1080'

OPENPILOT_FONT = str(Path(BASEDIR, 'selfdrive/assets/fonts/Inter-Regular.ttf').resolve())

logger = logging.getLogger('clip.py')

# Initialize cffi for OpenGL calls
_ffi = cffi.FFI()
_ffi.cdef("""
  void glReadPixels(int x, int y, int width, int height, unsigned int format, unsigned int type, void *data);
  void glBindFramebuffer(unsigned int target, unsigned int framebuffer);
""")
# Load OpenGL library explicitly (libGL.so on Linux)
import ctypes.util
if platform.system() == 'Linux':
  opengl_lib = ctypes.util.find_library('GL') or 'libGL.so.1'
  _opengl = _ffi.dlopen(opengl_lib)
else:
  _opengl = _ffi.dlopen(None)


def extract_frame_from_texture(render_texture: rl.RenderTexture, width: int, height: int) -> bytes:
  """Extract RGB24 pixel data from a RenderTexture.

  Args:
    render_texture: The RenderTexture to read from
    width: Width of the texture
    height: Height of the texture

  Returns:
    RGB24 pixel data as bytes (width * height * 3 bytes)
  """
  # Bind the framebuffer to read from it
  _opengl.glBindFramebuffer(0x8D40, render_texture.id)  # GL_FRAMEBUFFER = 0x8D40

  # Allocate buffer for RGBA data (OpenGL returns RGBA)
  rgba_size = width * height * 4
  rgba_buffer = _ffi.new("unsigned char[]", rgba_size)

  # Read pixels from framebuffer (RGBA format)
  _opengl.glReadPixels(0, 0, width, height, 0x1908, 0x1401, rgba_buffer)  # GL_RGBA = 0x1908, GL_UNSIGNED_BYTE = 0x1401

  # Unbind framebuffer
  _opengl.glBindFramebuffer(0x8D40, 0)

  # Convert RGBA to RGB24 and flip vertically using numpy (much faster)
  rgba_array = np.frombuffer(_ffi.buffer(rgba_buffer), dtype=np.uint8).reshape(height, width, 4)
  # Extract RGB channels and flip vertically in one operation
  rgb_array = rgba_array[::-1, :, :3].reshape(height * width * 3)
  return rgb_array.tobytes()


def escape_ffmpeg_text(value: str):
  special_chars = {',': '\\,', ':': '\\:', '=': '\\=', '[': '\\[', ']': '\\]'}
  value = value.replace('\\', '\\\\\\\\\\\\\\\\')
  for char, escaped in special_chars.items():
    value = value.replace(char, escaped)
  return value


def get_logreader(route: Route, start: int, end: int):
  """Get LogReader for the route, loading all segments that contain the time range."""
  # Calculate which segments we need (each segment is ~60 seconds)
  start_seg = start // 60
  end_seg = end // 60

  # Load all segments from start_seg to end_seg
  # Use route identifier with segment range syntax
  if start_seg == end_seg:
    # Single segment
    route_id = f"{route.name.canonical_name}/{start_seg}"
  else:
    # Multiple segments
    route_id = f"{route.name.canonical_name}/{start_seg}:{end_seg + 1}"

  return LogReader(route_id, default_mode=ReadMode.RLOG)


def get_meta_text(lr: LogReader, route: Route):
  init_data = lr.first('initData')
  car_params = lr.first('carParams')
  origin_parts = init_data.gitRemote.split('/')
  origin = origin_parts[3] if len(origin_parts) > 3 else 'unknown'
  return ', '.join([
    f"openpilot v{init_data.version}",
    f"route: {route.name.canonical_name}",
    f"car: {car_params.carFingerprint}",
    f"origin: {origin}",
    f"branch: {init_data.gitBranch}",
    f"commit: {init_data.gitCommit[:7]}",
    f"modified: {str(init_data.dirty).lower()}",
  ])


def parse_args(parser: ArgumentParser):
  args = parser.parse_args()
  if args.demo:
    args.route = DEMO_ROUTE
    if args.start is None or args.end is None:
      args.start = DEMO_START
      args.end = DEMO_END
  elif args.route.count('/') == 1:
    if args.start is None or args.end is None:
      parser.error('must provide both start and end if timing is not in the route ID')
  elif args.route.count('/') == 3:
    if args.start is not None or args.end is not None:
      parser.error('don\'t provide timing when including it in the route ID')
    parts = args.route.split('/')
    args.route = '/'.join(parts[:2])
    args.start = int(parts[2])
    args.end = int(parts[3])
  if args.end <= args.start:
    parser.error(f'end ({args.end}) must be greater than start ({args.start})')

  try:
    args.route = Route(args.route, data_dir=args.data_dir)
  except Exception as e:
    parser.error(f'failed to get route: {e}')

  # Calculate route length (approximate: 60 seconds per segment)
  length = round(args.route.max_seg_number * 60)
  if args.start >= length:
    parser.error(f'start ({args.start}s) cannot be after end of route ({length}s)')
  if args.end > length:
    parser.error(f'end ({args.end}s) cannot be after end of route ({length}s)')

  return args


def populate_car_params(lr: LogReader):
  init_data = lr.first('initData')
  assert init_data is not None

  params = Params()
  entries = init_data.params.entries
  for cp in entries:
    key, value = cp.key, cp.value
    try:
      params.put(key, params.cpp2python(key, value))
    except UnknownKeyName:
      # forks of openpilot may have other Params keys configured. ignore these
      logger.warning(f"unknown Params key '{key}', skipping")
  logger.debug('persisted CarParams')


def validate_env(parser: ArgumentParser):
  # Check ffmpeg
  if shutil.which('ffmpeg') is None:
    parser.exit(1, 'clip.py: error: missing ffmpeg command, is it installed?\n')
  # Check Xvfb (needed for GLFW to create OpenGL context on Linux)
  if platform.system() == 'Linux' and shutil.which('Xvfb') is None:
    parser.exit(1, 'clip.py: error: missing Xvfb command, is it installed?\n')


def validate_output_file(output_file: str):
  if not output_file.endswith('.mp4'):
    raise ArgumentTypeError('output must be an mp4')
  return output_file


def validate_route(route: str):
  if route.count('/') not in (1, 3):
    raise ArgumentTypeError(f'route must include or exclude timing, example: {DEMO_ROUTE}')
  return route


def validate_title(title: str):
  if len(title) > 80:
    raise ArgumentTypeError('title must be no longer than 80 chars')
  return title


def get_frame_reader(route: Route, quality: Literal['low', 'high']):
  """Get FrameReader for the appropriate camera stream."""
  if quality == 'low':
    camera_paths = route.qcamera_paths()
  else:
    camera_paths = route.camera_paths()

  # Filter out None values and get first valid path
  camera_paths = [p for p in camera_paths if p is not None]
  if not camera_paths:
    raise ValueError(f'No camera files found for route {route.name.canonical_name}')

  return FrameReader(camera_paths[0], pix_fmt='nv12')


def clip(
  quality: Literal['low', 'high'],
  prefix: str,
  route: Route,
  out: str,
  start: int,
  end: int,
  speed: int,
  target_mb: int,
  title: str | None,
):
  logger.info(f'clipping route {route.name.canonical_name}, start={start} end={end} quality={quality} target_filesize={target_mb}MB')
  lr = get_logreader(route, start, end)

  duration = end - start
  bit_rate_kbps = int(round(target_mb * 8 * 1024 * 1024 / duration / 1000))

  # Parse resolution
  width, height = map(int, RESOLUTION.split('x'))

  box_style = 'box=1:boxcolor=black@0.33:boxborderw=7'
  meta_text = get_meta_text(lr, route)
  overlays = [
    # metadata overlay
    f"drawtext=text='{escape_ffmpeg_text(meta_text)}':fontfile={OPENPILOT_FONT}:fontcolor=white:fontsize=15:{box_style}:x=(w-text_w)/2:y=5.5:enable='between(t,1,5)'",
    # route time overlay
    f"drawtext=text='%{{eif\\:floor(({start}+t)/60)\\:d\\:2}}\\:%{{eif\\:mod({start}+t\\,60)\\:d\\:2}}':fontfile={OPENPILOT_FONT}:fontcolor=white:fontsize=24:{box_style}:x=w-text_w-38:y=38"
  ]
  if title:
    overlays.append(f"drawtext=text='{escape_ffmpeg_text(title)}':fontfile={OPENPILOT_FONT}:fontcolor=white:fontsize=32:{box_style}:x=(w-text_w)/2:y=53")

  if speed > 1:
    overlays += [
      f"setpts=PTS/{speed}",
      "fps=60",
    ]

  # ffmpeg command using rawvideo input from stdin
  ffmpeg_cmd = [
    'ffmpeg', '-y',
    '-f', 'rawvideo',
    '-pix_fmt', 'rgb24',
    '-s', RESOLUTION,
    '-r', str(FRAMERATE),
    '-i', 'pipe:0',
    '-c:v', 'libx264',
    '-maxrate', f'{bit_rate_kbps}k',
    '-bufsize', f'{bit_rate_kbps*2}k',
    '-crf', '23',
    '-filter:v', ','.join(overlays),
    '-preset', 'ultrafast',
    '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart',
    '-f', 'mp4',
    '-t', str(duration),
    out,
  ]

  # Set up prefix and shared download cache
  os.environ['OPENPILOT_PREFIX'] = prefix
  from openpilot.system.hardware.hw import DEFAULT_DOWNLOAD_CACHE_ROOT
  os.environ['COMMA_CACHE'] = DEFAULT_DOWNLOAD_CACHE_ROOT

  # Populate car params
  populate_car_params(lr)

  # Setup Xvfb for GLFW (Linux only - GLFW needs X11 display for OpenGL context)
  # MUST set DISPLAY before importing GUI components (GLFW reads DISPLAY at import time)
  xvfb_proc = None
  original_display = os.environ.get('DISPLAY')
  if platform.system() == 'Linux':
    # Check if existing DISPLAY is valid, create Xvfb if needed
    display = os.environ.get('DISPLAY')
    if not display or Popen(['xdpyinfo', '-display', display], stdout=DEVNULL, stderr=DEVNULL).wait() != 0:
      display = f':{randint(99, 999)}'
      xvfb_proc = Popen(['Xvfb', display, '-screen', '0', f'{width}x{height}x24'], stdout=DEVNULL, stderr=DEVNULL)
      # Wait for Xvfb to be ready (max 5s)
      for _ in range(50):
        if xvfb_proc.poll() is not None:
          raise RuntimeError(f'Xvfb failed to start (exit code {xvfb_proc.returncode})')
        if Popen(['xdpyinfo', '-display', display], stdout=DEVNULL, stderr=DEVNULL).wait() == 0:
          break
        time.sleep(0.1)
      else:
        raise RuntimeError('Xvfb failed to become ready within 5s')
    os.environ['DISPLAY'] = display

  env = os.environ.copy()

  # Import GUI components AFTER DISPLAY is set (GLFW reads DISPLAY at import time)
  from openpilot.system.ui.lib.application import gui_app
  from openpilot.selfdrive.ui.layouts.main import MainLayout
  from openpilot.selfdrive.ui.ui_state import ui_state, device

  try:
    # Load frames and setup VisionIPC
    logger.info('loading camera frames...')
    fr = get_frame_reader(route, quality)

    # Create VisionIpcServer for camera frames
    vipc_server = VisionIpcServer("camerad")
    vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 20, fr.w, fr.h)
    vipc_server.start_listener()

    # Small delay to let VisionIPC server start
    time.sleep(0.1)

    # Load and filter log messages to time range
    logger.info('loading log messages...')
    messages = []
    route_start_timestamp = None

    # Get route start timestamp from first segment (for absolute time calculation)
    route_start_lr = LogReader(f"{route.name.canonical_name}/0", default_mode=ReadMode.RLOG)
    for msg in route_start_lr:
      route_start_timestamp = msg.logMonoTime
      break  # Just need the first message's timestamp

    # Load all messages from the relevant segments
    for msg in lr:
      messages.append(msg)

    # Filter messages to time range (start to end in seconds relative to route start)
    if route_start_timestamp is None:
      # Fallback: use first message timestamp if we couldn't get route start
      route_start_timestamp = messages[0].logMonoTime if messages else 0

    start_mono_time = route_start_timestamp + int(start * 1e9)
    end_mono_time = route_start_timestamp + int(end * 1e9)
    filtered_messages = [
      msg for msg in messages
      if start_mono_time <= msg.logMonoTime <= end_mono_time
    ]

    if not filtered_messages:
      raise ValueError(f'No messages found in time range {start}s to {end}s (route start timestamp: {route_start_timestamp})')

    # Services that ui_state.sm needs
    needed_services = set(ui_state.sm.services)

    # Initialize Python UI in headless mode
    logger.debug('initializing UI...')
    # Set environment to force headless/offscreen rendering
    os.environ.setdefault('HEADLESS', '1')

    # Force render texture creation by setting scale != 1.0
    original_scale = os.environ.pop('SCALE', None)
    os.environ['SCALE'] = '2.0'

    # Initialize window and create render texture
    gui_app.init_window("Clip Renderer", fps=FRAMERATE)
    if gui_app._render_texture is not None:
      rl.unload_render_texture(gui_app._render_texture)
    gui_app._render_texture = rl.load_render_texture(width, height)
    rl.set_texture_filter(gui_app._render_texture.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

    # Initialize MainLayout
    main_layout = MainLayout()
    main_layout.set_rect(rl.Rectangle(0, 0, width, height))

    # Restore original scale
    if original_scale:
      os.environ['SCALE'] = original_scale
    else:
      os.environ.pop('SCALE', None)

    # Start ffmpeg with stdin pipe
    logger.info(f'recording in progress ({duration}s)...')
    ffmpeg_proc = Popen(ffmpeg_cmd, stdin=PIPE, env=env)

    try:
      frame_count = 0
      target_frames = int(duration * FRAMERATE)
      frame_time = 1.0 / FRAMERATE
      frame_time_ns = int(1e9 / FRAMERATE)
      start_time = time.monotonic()
      current_frame_time = start_mono_time
      msg_idx = 0
      last_seen_frame_id = 0

      # Render loop
      while frame_count < target_frames and msg_idx < len(filtered_messages):
        # Collect all messages for this frame time window
        frame_end_time = current_frame_time + frame_time_ns
        frame_messages = []
        current_frame_id = last_seen_frame_id  # Start with last seen frame

        while msg_idx < len(filtered_messages):
          msg = filtered_messages[msg_idx]
          if msg.logMonoTime > frame_end_time:
            break

          # Collect messages for SubMaster
          if msg.which() in needed_services:
            frame_messages.append(msg)

          # Track roadEncodeIdx to get current frame
          if msg.which() == 'roadEncodeIdx':
            eidx = msg.roadEncodeIdx
            # Type.fullHEVC = 1 (from cereal/log.capnp)
            if eidx.type == 1:  # fullHEVC
              current_frame_id = eidx.frameId
              last_seen_frame_id = current_frame_id

          msg_idx += 1

        # Feed messages to SubMaster
        if frame_messages:
          ui_state.sm.update_msgs(time.monotonic(), frame_messages)

        # Update UI state (manually call update methods since we're not using ZMQ)
        ui_state._update_state()
        ui_state._update_status()
        if time.monotonic() - ui_state._param_update_time > 5.0:
          ui_state.update_params()
        device.update()

        # Get camera frame and send to VisionIPC
        if current_frame_id >= 0 and fr.frame_count > 0:
          try:
            frame_idx = min(current_frame_id, fr.frame_count - 1)
            if 0 <= frame_idx < fr.frame_count:
              frame_data = fr.get(frame_idx)
              # Convert numpy array to bytes
              frame_bytes = frame_data.tobytes() if isinstance(frame_data, np.ndarray) else frame_data
              # Send frame to VisionIPC
              vipc_server.send(VisionStreamType.VISION_STREAM_ROAD, frame_bytes,
                             frame_id=current_frame_id,
                             timestamp_sof=current_frame_time,
                             timestamp_eof=current_frame_time)
          except (KeyError, IndexError, ValueError) as e:
            logger.warning(f'Failed to get frame {current_frame_id}: {e}')

        # Render frame to texture
        rl.begin_texture_mode(gui_app._render_texture)
        rl.clear_background(rl.BLACK)
        main_layout.render()
        rl.end_texture_mode()

        # Extract frame pixels
        frame_data = extract_frame_from_texture(gui_app._render_texture, width, height)

        # Write to ffmpeg
        assert ffmpeg_proc.stdin is not None
        ffmpeg_proc.stdin.write(frame_data)
        ffmpeg_proc.stdin.flush()

        frame_count += 1
        current_frame_time += frame_time_ns

        # Rate limiting to match FRAMERATE
        next_frame_time = (frame_count + 1) * frame_time
        sleep_time = next_frame_time - (time.monotonic() - start_time)
        if sleep_time > 0:
          time.sleep(sleep_time)

      # Cleanup
      assert ffmpeg_proc.stdin is not None
      ffmpeg_proc.stdin.close()
      ffmpeg_proc.wait()

      if ffmpeg_proc.returncode != 0:
        raise ChildProcessError(f'ffmpeg failed with exit code {ffmpeg_proc.returncode}')

    finally:
      # Cleanup UI
      gui_app.close()
      del vipc_server

    logger.info(f'recording complete: {Path(out).resolve()}')
  finally:
    # Cleanup Xvfb and restore DISPLAY
    if xvfb_proc is not None:
      xvfb_proc.terminate()
      xvfb_proc.wait()
    # Restore original DISPLAY
    if original_display:
      os.environ['DISPLAY'] = original_display
    else:
      os.environ.pop('DISPLAY', None)


def main():
  p = ArgumentParser(prog='clip.py', description='clip your openpilot route.', epilog='comma.ai')
  validate_env(p)
  route_group = p.add_mutually_exclusive_group(required=True)
  route_group.add_argument('route', nargs='?', type=validate_route, help=f'The route (e.g. {DEMO_ROUTE} or {DEMO_ROUTE}/{DEMO_START}/{DEMO_END})')
  route_group.add_argument('--demo', help='use the demo route', action='store_true')
  p.add_argument('-d', '--data-dir', help='local directory where route data is stored')
  p.add_argument('-e', '--end', help='stop clipping at <end> seconds', type=int)
  p.add_argument('-f', '--file-size', help='target file size (Discord/GitHub support max 10MB, default is 9MB)', type=float, default=9.)
  p.add_argument('-o', '--output', help='output clip to (.mp4)', type=validate_output_file, default=DEFAULT_OUTPUT)
  p.add_argument('-p', '--prefix', help='openpilot prefix', default=f'clip_{randint(100, 99999)}')
  p.add_argument('-q', '--quality', help='quality of camera (low = qcam, high = hevc)', choices=['low', 'high'], default='high')
  p.add_argument('-x', '--speed', help='record the clip at this speed multiple', type=int, default=1)
  p.add_argument('-s', '--start', help='start clipping at <start> seconds', type=int)
  p.add_argument('-t', '--title', help='overlay this title on the video (e.g. "Chill driving across the Golden Gate Bridge")', type=validate_title)
  args = parse_args(p)
  exit_code = 1
  try:
    clip(
      quality=args.quality,
      prefix=args.prefix,
      route=args.route,
      out=args.output,
      start=args.start,
      end=args.end,
      speed=args.speed,
      target_mb=args.file_size,
      title=args.title,
    )
    exit_code = 0
  except KeyboardInterrupt as e:
    logger.exception('interrupted by user', exc_info=e)
  except Exception as e:
    logger.exception('encountered error', exc_info=e)
  sys.exit(exit_code)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
  main()
