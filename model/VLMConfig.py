from .LMConfig import LMConfig
from typing import List


class VLMConfig(LMConfig):
    model_type = "minimind-v"

    def __init__(
            self,
            image_special_token: str = '@' * 196,
            image_ids: List = [34] * 196,
            **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        super().__init__(**kwargs)
