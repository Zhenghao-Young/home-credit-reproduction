class OfflineContextParams:
    pass


class Context:
    def __init__(self):
        self.params = OfflineContextParams()

    def channel_send(self, *args, **kwargs):
        return None
