class _Config:
    def __init__(self):
        self.image_size = 224
        self.vocab_path = ''

    def update(self, cfg):
        for k in cfg.__dict__:
            if k not in self.__dict__:
                setattr(self, k, cfg.__dict__[k])

cfg = _Config()