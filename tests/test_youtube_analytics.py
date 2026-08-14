from shorts_factory.providers.youtube_analytics import YouTubeAnalyticsProvider


class Executable:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class Reports:
    def query(self, **kwargs):
        if kwargs.get("dimensions") == "elapsedVideoTimeRatio":
            return Executable({"rows": [[0.0, 1.0], [0.5, 0.72], [1.0, 0.41]]})
        return Executable({"rows": [[100, 50, 30, 66.7, 3, 1]]})


class Service:
    def reports(self):
        return Reports()


def test_fetches_basic_metrics_and_retention_curve(tmp_path):
    secrets = tmp_path / "client.json"
    secrets.write_text("{}")
    provider = YouTubeAnalyticsProvider(str(secrets), str(tmp_path / "token.json"))
    provider._service = Service()
    metrics = provider.get_video_metrics("abc", "2026-01-01", "2026-01-02")
    assert metrics.views == 100
    assert metrics.subscribers_gained == 3
    assert metrics.retention_curve[1] == {
        "elapsed_video_time_ratio": 0.5,
        "audience_watch_ratio": 0.72,
    }
