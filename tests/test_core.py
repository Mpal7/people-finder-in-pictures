"""Small dependency-light tests for the pure helpers.

Run with:  python -m unittest discover -s tests -v
(These do not import FastAPI, InsightFace or ONNX Runtime.)
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pick_reference_faces import stage_apply
from photo_finder import review
from photo_finder.imaging import image_date, load_dup_map
from photo_finder.jobs import JobManager, pid_alive
from photo_finder.paths import is_within, rel_parts, safe_join
from photo_finder.similarity import dhash, hamming, near_duplicate_groups
from photo_finder.thumbs import ThumbCache


def _pattern(seed: int = 0, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Smooth random blocks: survives JPEG recompression well.
    small = rng.integers(0, 255, size=(12, 12, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((size, size), Image.BILINEAR)
    return np.asarray(img)


class PathTests(unittest.TestCase):
    def test_relative_paths_accept_both_separators(self):
        self.assertEqual(rel_parts(r"2024\July\a.jpg"), ("2024", "July", "a.jpg"))
        self.assertEqual(rel_parts("2024/July/a.jpg"), ("2024", "July", "a.jpg"))
        self.assertEqual(rel_parts("./2024//a.jpg"), ("2024", "a.jpg"))

    def test_absolute_and_traversal_paths_are_rejected(self):
        for bad in ("/etc/passwd", r"\Windows\a.jpg", r"C:\Photos\a.jpg",
                    "C:relative.jpg", r"2024\C:relative.jpg",
                    r"\\server\share\a.jpg", "../a.jpg",
                    r"2024\..\..\a.jpg", ""):
            with self.subTest(path=bad):
                with self.assertRaises(ValueError):
                    rel_parts(bad)

    def test_safe_join_stays_under_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertEqual(safe_join(root, r"2024\a.jpg"), root / "2024" / "a.jpg")
            with self.assertRaises(ValueError):
                safe_join(root, r"\outside.jpg")

    def test_is_within(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertTrue(is_within(root / "a" / "b.jpg", root))
            self.assertTrue(is_within(root, root))
            self.assertFalse(is_within(root.parent, root))

    def test_current_process_is_alive(self):
        self.assertTrue(pid_alive(os.getpid()))


class JobManagerTests(unittest.TestCase):
    def test_stopping_job_becomes_stopped(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            data.mkdir(exist_ok=True)
            (data / "jobs").mkdir(exist_ok=True)
            job = {
                "id": "20260101-000000-index", "kind": "index", "title": "test",
                "cmd": [], "pid": 99_999_999, "started": "2026-01-01 00:00:00",
                "ended": None, "status": "stopping",
                "log": str(data / "jobs" / "missing.log"),
                "progress": str(data / "jobs" / "missing.progress.json"),
                "meta": {},
            }
            (data / "jobs.json").write_text(json.dumps([job]), encoding="utf-8")
            manager = JobManager(data)
            self.assertEqual(manager.jobs[0]["status"], "stopped")
            self.assertTrue(manager.jobs[0]["ended"])


class SimilarityTests(unittest.TestCase):
    def test_dhash_survives_recompression(self):
        arr = _pattern(1)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.jpg"
            Image.fromarray(arr).save(p, quality=55)
            arr2 = np.asarray(Image.open(p).convert("RGB"))
        self.assertLessEqual(hamming(dhash(arr), dhash(arr2)), 4)

    def test_clustering(self):
        a = 0xAAAAAAAAAAAAAAAA
        b = a ^ 0b111111          # distance 6
        c = 0x5555555555555555    # distance 64 from both
        gid, rank, size = near_duplicate_groups([a, b, c], [1.5, 1.5, 1.5], threshold=6)
        self.assertEqual(gid[0], gid[1])
        self.assertNotEqual(gid[0], gid[2])
        self.assertEqual((rank[0], rank[1], rank[2]), (0, 1, 0))
        self.assertEqual((size[0], size[1], size[2]), (2, 2, 1))

    def test_aspect_ratio_blocks_grouping(self):
        a = 0xAAAAAAAAAAAAAAAA
        b = a ^ 0b1
        gid, _, size = near_duplicate_groups([a, b], [1.0, 2.0], threshold=6)
        self.assertNotEqual(gid[0], gid[1])
        self.assertEqual(size, [1, 1])

    def test_disabled_grouping(self):
        gid, rank, size = near_duplicate_groups([0, 0], [1, 1], threshold=0)
        self.assertEqual(gid, [0, 1])
        self.assertEqual(rank, [0, 0])
        self.assertEqual(size, [1, 1])


class DuplicateMapTests(unittest.TestCase):
    def test_folder_rank_picks_canonical_copy(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "duplicates.csv"
            csv_path.write_text(
                "group_id,relpath\n"
                "1,camera/aaa.jpg\n"
                "1,phone/zzz.jpg\n",
                encoding="utf-8",
            )
            # Alphabetically camera/aaa would win; the rank map must override it.
            by_path = load_dup_map(csv_path, {"camera": 10, "phone": 80})
            self.assertFalse(by_path["camera/aaa.jpg"][1])
            self.assertTrue(by_path["phone/zzz.jpg"][1])


class DateTests(unittest.TestCase):
    def test_exif_date(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.jpg"
            exif = Image.Exif()
            exif[36867] = "2021:07:04 12:30:00"
            Image.new("RGB", (16, 16), "red").save(p, exif=exif)
            self.assertEqual(image_date(p), "2021-07-04")

    def test_mtime_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "e.jpg"
            Image.new("RGB", (16, 16), "blue").save(p)
            stamp = time.mktime((2019, 2, 3, 10, 0, 0, 0, 0, -1))
            os.utime(p, (stamp, stamp))
            self.assertEqual(image_date(p), "2019-02-03")

    def test_invalid_date_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "f.jpg"
            exif = Image.Exif()
            exif[36867] = "0000:00:00 00:00:00"
            Image.new("RGB", (16, 16), "green").save(p, exif=exif)
            stamp = time.mktime((2020, 1, 2, 0, 0, 0, 0, 0, -1))
            os.utime(p, (stamp, stamp))
            self.assertEqual(image_date(p), "2020-01-02")


class ReviewGenerateTests(unittest.TestCase):
    def test_groups_dates_and_csv(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "photos"
            root.mkdir()
            arr = _pattern(3, 120)
            Image.fromarray(arr).save(root / "a.jpg", quality=90)
            Image.fromarray(arr).save(root / "b.jpg", quality=45)
            Image.fromarray(_pattern(4, 120)).save(root / "c.jpg", quality=90)
            candidates = [
                {"relpath": "a.jpg", "score": 10.0, "per_model": {"w600k": 0.8},
                 "faces": [{"bbox": [10, 10, 50, 50], "score": 10.0}]},
                {"relpath": "b.jpg", "score": 9.0, "per_model": {"w600k": 0.7},
                 "faces": [{"bbox": [12, 12, 52, 52], "score": 9.0}]},
                {"relpath": "c.jpg", "score": 5.0, "per_model": {"w600k": 0.4},
                 "faces": [{"bbox": [20, 20, 60, 60], "score": 5.0}]},
            ]
            out = Path(td) / "run"
            review.generate(out, candidates, ["r.jpg"], ["w600k"], root,
                            lambda p, max_side: np.asarray(Image.open(p).convert("RGB")),
                            near_dup_threshold=6)
            data = json.loads((out / "data.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["images"]), 3)
            self.assertEqual(data["images"][0]["g"], data["images"][1]["g"])
            self.assertNotEqual(data["images"][0]["g"], data["images"][2]["g"])
            self.assertEqual(data["images"][0]["g_size"], 2)
            self.assertEqual(data["images"][2]["g_size"], 1)
            self.assertTrue(data["images"][0]["date"])
            html = (out / "index.html").read_text(encoding="utf-8")
            self.assertIn("group similar", html)
            self.assertIn("similar", html)
            csv_text = (out / "candidates.csv").read_text(encoding="utf-8")
            self.assertIn("near_group_rank", csv_text)
            self.assertIn("shoot_date", csv_text)


class PickerApplyTests(unittest.TestCase):
    def test_sidecar_and_same_stem_collision(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            dest = root / "refs"
            sel = root / "selection.json"
            src.mkdir()
            Image.new("RGB", (200, 100), "red").save(src / "same.jpg")
            Image.new("RGB", (200, 100), "blue").save(src / "same.png")
            kps = [[0.3, 0.4], [0.7, 0.4], [0.5, 0.55], [0.35, 0.7], [0.65, 0.7]]
            payload = {"selections": [
                {"file": "same.jpg", "face": 0, "bbox_norm": [0.2, 0.2, 0.8, 0.8],
                 "kps_norm": kps},
            ]}
            sel.write_text(json.dumps(payload), encoding="utf-8")
            stage_apply(sel, src, dest)
            self.assertTrue((dest / "same_face.jpg").exists())
            sidecar = json.loads((dest / "same_face.jpg.kps.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["src"], "same.jpg")
            self.assertIn("kps_norm", sidecar)

            # Re-applying the same selection is idempotent (no hash-suffixed copy).
            stage_apply(sel, src, dest)
            self.assertEqual(len(list(dest.glob("*.jpg"))), 1)

            # A different source with the same stem gets a deterministic suffix.
            payload["selections"].append(
                {"file": "same.png", "face": 0, "bbox_norm": [0.2, 0.2, 0.8, 0.8],
                 "kps_norm": kps})
            sel.write_text(json.dumps(payload), encoding="utf-8")
            stage_apply(sel, src, dest)
            self.assertEqual(len(list(dest.glob("*.jpg"))), 2)
            self.assertEqual(len(list(dest.glob("same_*_face.jpg"))), 1)


class ThumbCacheTests(unittest.TestCase):
    def test_eviction_and_clear(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            src.mkdir()
            cache = ThumbCache(root / "cache", max_bytes=2500, max_age_s=86400)
            for i in range(8):
                p = src / f"{i}.jpg"
                Image.fromarray(_pattern(i, 160)).save(p, quality=90)
                cache.path_for(p, p.name, 64)
            stats = cache.stats()
            self.assertLessEqual(stats["bytes"], 2500)
            self.assertGreater(stats["files"], 0)
            removed = cache.clear()
            self.assertEqual(removed, stats["files"])
            self.assertEqual(cache.stats()["files"], 0)


if __name__ == "__main__":
    unittest.main()
