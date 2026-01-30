class ExperimentLogger:
    def __init__(self, enabled=False):
        self.enabled = enabled

    def log_step(self, **data):
        if not self.enabled:
            return
        # Placeholder hook for future structured logging.

    def close(self):
        pass
