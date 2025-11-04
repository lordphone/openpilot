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

  # Patch messaging to work without real sockets (no ui_replay subprocess needed)
  # Monkey-patch sub_sock before any imports try to use it
  import msgq
  original_sub_sock = msgq.sub_sock
  def mock_sub_sock(endpoint, *args, **kwargs):
    """Mock socket creation for replay mode - returns a dummy object."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.receive = MagicMock(return_value=None)
    return mock
  msgq.sub_sock = mock_sub_sock

  # Import GUI components AFTER DISPLAY is set (GLFW reads DISPLAY at import time)
  from openpilot.system.ui.lib.application import gui_app
  from openpilot.selfdrive.ui.layouts.main import MainLayout
  from openpilot.selfdrive.ui.ui_state import ui_state, device

  try:
    # Load frames and setup VisionIPC
    logger.info('loading camera frames...')
    fr = get_frame_reader(route, quality)
    logger.info(f'Camera frames loaded: count={fr.frame_count}, dimensions={fr.w}x{fr.h}')

    # Test reading a few frames
    if fr.frame_count > 0:
      try:
        test_frame_0 = fr.get(0)
        test_frame_mid = fr.get(min(100, fr.frame_count - 1))
        logger.info(f'Frame 0: shape={test_frame_0.shape if hasattr(test_frame_0, "shape") else type(test_frame_0)}, size={len(test_frame_0.tobytes() if hasattr(test_frame_0, "tobytes") else test_frame_0)} bytes')
        logger.info(f'Frame {min(100, fr.frame_count - 1)}: shape={test_frame_mid.shape if hasattr(test_frame_mid, "shape") else type(test_frame_mid)}, size={len(test_frame_mid.tobytes() if hasattr(test_frame_mid, "tobytes") else test_frame_mid)} bytes')
      except Exception as e:
        logger.warning(f'Failed to read test frames: {e}')

    # Create shared memory directory for VisionIPC
    shm_dir = Path(f'/dev/shm/{prefix}')
    shm_dir.mkdir(parents=True, exist_ok=True)

    # Create VisionIpcServer for camera frames
    vipc_server = VisionIpcServer("camerad")
    vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 20, fr.w, fr.h)
    vipc_server.start_listener()

    # Small delay to let VisionIPC server start
    time.sleep(0.1)

    # Load log messages
    logger.info('loading log messages...')
    filtered_messages = []
    start_mono_time = None
    message_types = {}
    road_encode_messages = []

    # Load all messages from the segments (LogReader already loaded correct segments)
    for msg in lr:
      if start_mono_time is None:
        start_mono_time = msg.logMonoTime
      filtered_messages.append(msg)

      # Count message types
      msg_type = msg.which()
      message_types[msg_type] = message_types.get(msg_type, 0) + 1

      # Track roadEncodeIdx messages
      if msg_type == 'roadEncodeIdx':
        road_encode_messages.append((msg.logMonoTime, msg.roadEncodeIdx.frameId, msg.roadEncodeIdx.type))

    logger.info(f'Loaded {len(filtered_messages)} messages')
    logger.info(f'Message types: {dict(sorted(message_types.items(), key=lambda x: x[1], reverse=True)[:10])}')
    logger.info(f'First message timestamp: {start_mono_time}')
    logger.info(f'Last message timestamp: {filtered_messages[-1].logMonoTime if filtered_messages else "N/A"}')
    logger.info(f'Duration: {(filtered_messages[-1].logMonoTime - start_mono_time) / 1e9:.1f}s' if filtered_messages else 'N/A')
    logger.info(f'Found {len(road_encode_messages)} roadEncodeIdx messages')
    if road_encode_messages:
      logger.info(f'  First: timestamp={road_encode_messages[0][0]}, frameId={road_encode_messages[0][1]}, type={road_encode_messages[0][2]}')
      logger.info(f'  Last: timestamp={road_encode_messages[-1][0]}, frameId={road_encode_messages[-1][1]}, type={road_encode_messages[-1][2]}')
    if not filtered_messages:
      raise ValueError('No messages found in log')

    # Services that ui_state.sm needs
    needed_services = set(ui_state.sm.services)

    # Disable frequency tracking for replay mode (we're not real-time)
    ui_state.sm.simulation = True
    # Patch FrequencyTracker.valid to avoid ZeroDivisionError in replay mode
    from cereal.messaging import FrequencyTracker
    original_valid = FrequencyTracker.valid.fget
    def patched_valid(self):
      try:
        return original_valid(self)
      except ZeroDivisionError:
        return True  # Always valid in replay mode
    FrequencyTracker.valid = property(patched_valid)

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
      # Find first roadEncodeIdx message to align frame timing
      # Use timestampSof (camera frame timestamp) not logMonoTime for proper synchronization
      first_camera_frame_ts = None
      first_frame_id = None
      for msg in filtered_messages:
        if msg.which() == 'roadEncodeIdx' and msg.roadEncodeIdx.type == 1:  # fullHEVC
          first_camera_frame_ts = msg.roadEncodeIdx.timestampSof  # Use camera frame timestamp, not logMonoTime
          first_frame_id = msg.roadEncodeIdx.frameId
          logger.info(f'First camera frame: timestampSof={first_camera_frame_ts}, logMonoTime={msg.logMonoTime}, frameId={first_frame_id}')
          logger.info(f'  Offset: logMonoTime - timestampSof = {(msg.logMonoTime - msg.roadEncodeIdx.timestampSof) / 1e6:.1f}ms')
          break

      if first_camera_frame_ts is None:
        raise ValueError('No roadEncodeIdx messages found in log')

      logger.info(f'Start mono time: {start_mono_time}, offset: {(first_camera_frame_ts - start_mono_time) / 1e9:.1f}s')

      # Feed messages from first camera frame onwards to initialize ui_state properly
      # Match messages using camera frame timestampSof for proper synchronization
      all_needed_messages = [
        msg for msg in filtered_messages
        if msg.which() in needed_services and msg.logMonoTime >= first_camera_frame_ts
      ]
      if all_needed_messages:
        # Feed in batches to avoid overwhelming
        batch_size = 1000
        for i in range(0, len(all_needed_messages), batch_size):
          batch = all_needed_messages[i:i+batch_size]
          ui_state.sm.update_msgs(time.monotonic(), batch)
        # Update state after feeding all messages
        ui_state._update_state()
        ui_state._update_status()
        logger.info(f'Initialized ui_state with {len(all_needed_messages)} messages from first camera frame onwards')

      # Skip messages until we reach the first camera frame timestamp
      # This ensures UI state and camera frames are synchronized during rendering
      msg_idx = 0
      skipped_count = 0
      while msg_idx < len(filtered_messages) and filtered_messages[msg_idx].logMonoTime < first_camera_frame_ts:
        msg_idx += 1
        skipped_count += 1
      logger.info(f'Skipped {skipped_count} messages before first camera frame')

      # Iterate over camera frames from log (roadEncodeIdx messages)
      # This follows the pattern: for frame in log: ui.update_state(frame); ui.render(); capture_frame()
      camera_frames = [(msg.logMonoTime, msg.roadEncodeIdx.frameId, msg.roadEncodeIdx.timestampSof, msg.roadEncodeIdx.timestampEof)
                       for msg in filtered_messages
                       if msg.which() == 'roadEncodeIdx' and msg.roadEncodeIdx.type == 1]

      # Limit to duration
      end_time = first_camera_frame_ts + int(duration * 1e9)
      camera_frames = [(ts, fid, sof, eof) for ts, fid, sof, eof in camera_frames
                       if sof >= first_camera_frame_ts and sof <= end_time]

      logger.info(f'Found {len(camera_frames)} camera frames to render (duration: {duration}s)')

      # Count total messages in needed_services
      total_needed_messages = sum(1 for msg in filtered_messages if msg.which() in needed_services)
      logger.info(f'Total messages in needed_services: {total_needed_messages}')
      logger.info(f'Total camera frames: {len(camera_frames)}')
      logger.info(f'Expected: 1 camera frame per frame, multiple messages per frame')

      frame_count = 0
      msg_idx = 0  # Track position in filtered_messages for finding matching messages

      # Statistics tracking
      total_messages_processed = 0
      messages_by_frame = []
      message_counts_by_type = {}

      # Render loop: iterate over actual camera frames from log
      # Match original pattern: for frame in log: ui.update_state(frame); ui.render(); capture_frame()
      for frame_log_ts, frame_id, frame_timestamp_sof, frame_timestamp_eof in camera_frames:
        if frame_count >= int(duration * FRAMERATE):
          break  # Stop at target frame count

        # Feed messages that have arrived up to this camera frame's timestamp
        # Process messages chronologically, matching original replay behavior
        frame_messages = []
        while msg_idx < len(filtered_messages):
          msg = filtered_messages[msg_idx]
          # Stop when we reach messages after this camera frame
          if msg.logMonoTime > frame_timestamp_sof + 100e6:  # 100ms after camera frame (allow some latency)
            break
          if msg.which() in needed_services:
            frame_messages.append(msg)
            # Track message type counts
            msg_type = msg.which()
            message_counts_by_type[msg_type] = message_counts_by_type.get(msg_type, 0) + 1
          msg_idx += 1

        total_messages_processed += len(frame_messages)
        messages_by_frame.append((frame_count, len(frame_messages), frame_id, frame_timestamp_sof))

        # 1. Update UI state (equivalent to ui_state.update() in original)
        #    Original: ui_state.update() -> sm.update(0) reads from ZMQ
        #    Ours: Feed messages manually then update state
        if frame_messages:
          ui_state.sm.update_msgs(time.monotonic(), frame_messages)
          # Debug output every 60 frames
          if frame_count % 60 == 0:
            msg_types = [msg.which() for msg in frame_messages]
            logger.info(f'Frame {frame_count}: Processed {len(frame_messages)} messages: {set(msg_types)}')
        elif frame_count % 60 == 0:
          logger.warning(f'Frame {frame_count}: No messages matched to camera frame (frame_id={frame_id}, timestampSof={frame_timestamp_sof})')

        ui_state._update_state()
        ui_state._update_status()
        if time.monotonic() - ui_state._param_update_time > 5.0:
          ui_state.update_params()
        device.update()

        # 2. Send camera frame to VisionIPC (original: replay subprocess handles this)
        if fr.frame_count > 0:
          try:
            # Convert absolute frame ID to relative index for FrameReader
            frame_idx = frame_id - first_frame_id
            frame_idx = max(0, min(frame_idx, fr.frame_count - 1))  # Clamp to valid range

            frame_data = fr.get(frame_idx)
            # Convert numpy array to bytes
            frame_bytes = frame_data.tobytes() if isinstance(frame_data, np.ndarray) else frame_data

            # Debug output every 60 frames (3 seconds)
            if frame_count % 60 == 0:
              logger.info(f'Frame {frame_count}: frame_id={frame_id}, frame_idx={frame_idx}, frame_size={len(frame_bytes)} bytes')

            # Send frame to VisionIPC
            vipc_server.send(VisionStreamType.VISION_STREAM_ROAD, frame_bytes,
                           frame_id=frame_id,
                           timestamp_sof=frame_timestamp_sof,  # Use camera frame timestampSof
                           timestamp_eof=frame_timestamp_eof)  # Use camera frame timestampEof
          except (KeyError, IndexError, ValueError) as e:
            logger.warning(f'Failed to get frame {frame_id} (idx {frame_idx}): {e}')

        # 3. Render frame to texture (matches original exactly)
        rl.begin_texture_mode(gui_app._render_texture)
        rl.clear_background(rl.BLACK)
        main_layout.render()
        rl.end_texture_mode()

        # 4. Extract frame pixels (matches original exactly)
        frame_data = extract_frame_from_texture(gui_app._render_texture, width, height)

        # 5. Write to ffmpeg (matches original exactly, no rate limiting)
        assert ffmpeg_proc.stdin is not None
        ffmpeg_proc.stdin.write(frame_data)
        ffmpeg_proc.stdin.flush()

        frame_count += 1

      # Print comprehensive statistics
      logger.info('=' * 80)
      logger.info('MESSAGE PROCESSING STATISTICS:')
      logger.info(f'Total camera frames processed: {frame_count}')
      logger.info(f'Total messages processed: {total_messages_processed}')
      logger.info(f'Average messages per frame: {total_messages_processed / frame_count if frame_count > 0 else 0:.2f}')
      logger.info('')
      logger.info('Message counts by type:')
      for msg_type, count in sorted(message_counts_by_type.items(), key=lambda x: x[1], reverse=True):
        logger.info(f'  {msg_type}: {count}')
      logger.info('')
      logger.info('Messages per frame (first 10 and last 10):')
      for frame_num, msg_count, fid, ts in messages_by_frame[:10]:
        logger.info(f'  Frame {frame_num}: {msg_count} messages, frame_id={fid}, timestampSof={ts}')
      if len(messages_by_frame) > 20:
        logger.info('  ...')
        for frame_num, msg_count, fid, ts in messages_by_frame[-10:]:
          logger.info(f'  Frame {frame_num}: {msg_count} messages, frame_id={fid}, timestampSof={ts}')
      logger.info('')
      logger.info(f'Frames with 0 messages: {sum(1 for _, count, _, _ in messages_by_frame if count == 0)}')
      logger.info(f'Frames with messages: {sum(1 for _, count, _, _ in messages_by_frame if count > 0)}')
      logger.info('=' * 80)

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
