# -*- coding: utf-8 -*-
"""离线样本下载回归测试；不导入 transformers、不下载 CLIP 权重。"""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "multimodal_demo", ROOT / "docs/applied/systems/multimodal/code/demo.py"
)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)
NAMES = ("golden_retriever", "orange_cat", "red_car", "pizza")


def image_bytes(solid=False):
    image = Image.new("RGB", (16, 16), "gray")
    if not solid:
        image.paste("red", (0, 0, 8, 8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class BrokenResponse(io.BytesIO):
    def read(self, size=-1):
        if self.tell():
            raise TimeoutError("模拟读取中断")
        return super().read(10)


class SampleDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.images = Path(self.directory.name)
        self.samples = self.images / "samples"
        self.samples.mkdir()
        self.patch = patch.object(demo, "_IMAGES", str(self.images))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.good = image_bytes()
        self.solid = image_bytes(solid=True)

    def populate(self, payload):
        for name in NAMES:
            (self.samples / f"{name}.jpg").write_bytes(payload)

    def assert_no_temporary_files(self):
        self.assertEqual(list(self.samples.glob("*.tmp")), [])

    def test_validator_rejects_missing_corrupt_and_solid_images(self):
        path = self.samples / "test.jpg"
        self.assertFalse(demo.is_usable_sample(str(path)))
        for content in (b"not an image", self.solid, self.good[:50]):
            path.write_bytes(content)
            self.assertFalse(demo.is_usable_sample(str(path)))
        path.write_bytes(self.good)
        self.assertTrue(demo.is_usable_sample(str(path)))

    @patch("urllib.request.urlopen")
    def test_valid_cached_images_need_no_network(self, opener):
        self.populate(self.good)
        self.assertEqual(len(demo.download_sample_images()), 4)
        opener.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_missing_images_download_and_validate(self, opener):
        opener.side_effect = lambda *args, **kwargs: io.BytesIO(self.good)
        paths = demo.download_sample_images()
        self.assertEqual(len(paths), 4)
        self.assertTrue(all(demo.is_usable_sample(path) for path in paths))
        self.assertEqual(opener.call_count, 4)
        self.assertTrue(all(call.kwargs["timeout"] == 10 for call in opener.call_args_list))
        self.assert_no_temporary_files()

    @patch("urllib.request.urlopen")
    def test_invalid_existing_files_are_repaired(self, opener):
        for payload in (self.solid, b"broken"):
            with self.subTest(payload=payload[:8]):
                self.populate(payload)
                opener.side_effect = lambda *args, **kwargs: io.BytesIO(self.good)
                self.assertEqual(len(demo.download_sample_images()), 4)
                self.assertTrue(all(path.read_bytes() == self.good for path in self.samples.iterdir()))
                self.assert_no_temporary_files()

    @patch("urllib.request.urlopen")
    def test_invalid_responses_do_not_replace_existing_files(self, opener):
        self.populate(self.solid)
        for response in (b"<html>error</html>", self.solid, self.good[:50], b""):
            with self.subTest(response=response[:8]):
                opener.side_effect = lambda *args, **kwargs: io.BytesIO(response)
                self.assertEqual(demo.download_sample_images(), [])
                self.assertTrue(all(path.read_bytes() == self.solid for path in self.samples.iterdir()))
                self.assert_no_temporary_files()

    @patch("urllib.request.urlopen")
    def test_connection_failure_leaves_no_final_files(self, opener):
        opener.side_effect = URLError("模拟断网")
        self.assertEqual(demo.download_sample_images(), [])
        self.assertEqual(list(self.samples.iterdir()), [])

    @patch("urllib.request.urlopen")
    def test_partial_download_is_cleaned_and_can_retry(self, opener):
        opener.side_effect = lambda *args, **kwargs: BrokenResponse(self.good)
        self.assertEqual(demo.download_sample_images(), [])
        self.assertEqual(list(self.samples.iterdir()), [])
        opener.side_effect = lambda *args, **kwargs: io.BytesIO(self.good)
        self.assertEqual(len(demo.download_sample_images()), 4)
        self.assert_no_temporary_files()

    @patch("urllib.request.urlopen")
    def test_oversized_response_is_rejected(self, opener):
        opener.side_effect = lambda *args, **kwargs: io.BytesIO(b"x" * (10 * 1024 * 1024 + 1))
        self.assertEqual(demo.download_sample_images(), [])
        self.assertEqual(list(self.samples.iterdir()), [])

    @patch("time.monotonic")
    @patch("urllib.request.urlopen")
    def test_download_budget_exhaustion_is_cleaned(self, opener, monotonic):
        monotonic.side_effect = [value for _ in NAMES for value in (0, 31)]
        opener.side_effect = lambda *args, **kwargs: io.BytesIO(self.good)
        self.assertEqual(demo.download_sample_images(), [])
        self.assertEqual(list(self.samples.iterdir()), [])

    @patch.object(demo.os, "replace", side_effect=PermissionError("模拟替换失败"))
    @patch("urllib.request.urlopen")
    def test_replace_failure_preserves_old_file(self, opener, replace):
        self.populate(self.solid)
        opener.side_effect = lambda *args, **kwargs: io.BytesIO(self.good)
        self.assertEqual(demo.download_sample_images(), [])
        self.assertTrue(all(path.read_bytes() == self.solid for path in self.samples.iterdir()))
        self.assert_no_temporary_files()

    @patch("urllib.request.urlopen")
    def test_failed_sample_does_not_block_remaining_samples(self, opener):
        opener.side_effect = [URLError("首张失败")] + [io.BytesIO(self.good) for _ in range(3)]
        paths = demo.download_sample_images()
        self.assertEqual([Path(path).stem for path in paths], list(NAMES[1:]))
        self.assert_no_temporary_files()


if __name__ == "__main__":
    unittest.main()
