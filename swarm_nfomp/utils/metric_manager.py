class MetricManager:
    def __init__(self):
        self._metrics = {}
        self._iteration = 0

    def update_iteration(self):
        self._iteration += 1

    def add_metric(self, name, value, series_name="training"):
        if name not in self._metrics:
            self._metrics[name] = {}
        if series_name not in self._metrics[name]:
            self._metrics[name][series_name] = {}
        self._metrics[name][series_name][self._iteration] = value

    def log_metrics(self, logger):
        for name, series in self._metrics.items():
            for series_name, values in series.items():
                logger.report_scalar(title=name, series=series_name, value=values[self._iteration],
                                     iteration=self._iteration)
