import os
import pytest
import tempfile
from openpilot.tools.clip.run import clip, DEMO_ROUTE, DEMO_START
from openpilot.tools.lib.route import Route

class TestClip:
  @pytest.mark.slow
  def test_demo_clip(self):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
      # Use a very short duration for the test (e.g. 5 seconds)
      start = DEMO_START
      end = start + 5
      
      # Test with qcam and parallel download to ensure new features work
      clip(
        Route(DEMO_ROUTE),
        tmp.name,
        start=start,
        end=end,
        headless=True,
        use_qcam=True,
        parallel=True
      )
      
      assert os.path.exists(tmp.name)
      assert os.path.getsize(tmp.name) > 1024
