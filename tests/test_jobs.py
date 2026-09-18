"""Tests for app/jobs.py — background job registry (Phase 4)."""
import sys
import os
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from jobs import JobManager  # noqa: E402


class TestJobManager(unittest.TestCase):
    def setUp(self):
        self.mgr = JobManager()

    def test_start_and_finish_job(self):
        job = self.mgr.start_job("full_cycle")
        self.assertIsNotNone(job)
        self.assertTrue(self.mgr.has_active())

        self.mgr.mark_running(job.job_id)
        active = self.mgr.get_active()
        self.assertEqual(active.status, "running")
        self.assertIsNotNone(active.started_at)

        self.mgr.finish_job(job.job_id, success=True,
                            result_summary={"status": "success"})
        self.assertFalse(self.mgr.has_active())
        finished = self.mgr.get_job(job.job_id)
        self.assertEqual(finished.status, "done")
        self.assertEqual(finished.progress, 100)
        self.assertEqual(finished.result_summary["status"], "success")

    def test_rejects_overlap(self):
        first = self.mgr.start_job("full_cycle")
        self.assertIsNotNone(first)
        self.assertIsNone(self.mgr.start_job("full_cycle"))

    def test_progress_scaling(self):
        job = self.mgr.start_job("full_cycle")
        cb = self.mgr.make_progress_callback(job.job_id, base=5, span=90)
        cb(50, "halfway")
        active = self.mgr.get_active()
        self.assertEqual(active.progress, 50)
        self.assertEqual(active.message, "halfway")
        # clamp high
        cb(150, "over")
        self.assertEqual(active.progress, 95)

    def test_get_job_history(self):
        job = self.mgr.start_job("full_cycle")
        self.mgr.mark_running(job.job_id)
        self.mgr.finish_job(job.job_id, success=True)
        found = self.mgr.get_job(job.job_id)
        self.assertIsNotNone(found)
        self.assertEqual(found.status, "done")
        listing = self.mgr.list_jobs()
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0]["job_id"], job.job_id)

    def test_unknown_job(self):
        self.assertIsNone(self.mgr.get_job("nope"))

    def test_failed_job(self):
        job = self.mgr.start_job("full_cycle")
        self.mgr.finish_job(job.job_id, success=False, error="boom")
        found = self.mgr.get_job(job.job_id)
        self.assertEqual(found.status, "failed")
        self.assertEqual(found.error, "boom")
        self.assertFalse(self.mgr.has_active())

    def test_thread_safety(self):
        job = self.mgr.start_job("full_cycle")
        errors = []

        def update_many():
            for i in range(200):
                self.mgr.update_progress(job.job_id, i, f"step {i}")

        threads = [threading.Thread(target=update_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        active = self.mgr.get_active()
        self.assertEqual(active.progress, 100)


if __name__ == "__main__":
    unittest.main()
