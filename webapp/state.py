import threading

class AppState:
    def __init__(self):
        self.dataset = None
        self.base_model = None # The frozen Yamnet model
        self.classifier = None # The few-shot head
        self.none_bias_threshold = 0.0
        self.none_idx = None
        self.training_status = "Idle"
        self.training_log = []
        self.lock = threading.Lock()
        self.roc_curve = [] # List of (t, tpr, fpr)

    def log(self, message):
        with self.lock:
            self.training_log.append(message)
            # Keep log size manageable
            if len(self.training_log) > 1000:
                self.training_log.pop(0)

    def get_log(self):
        with self.lock:
            return "\n".join(self.training_log)

    def set_status(self, status):
        with self.lock:
            self.training_status = status

global_state = AppState()
