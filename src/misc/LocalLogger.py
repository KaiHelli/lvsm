import os
from shutil import copy2
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only

LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.experiment = None
        os.system(f"rm -r {LOG_PATH}")

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_video(self, key, videos, step=None, **kwargs):
        if not isinstance(videos, list):
            raise TypeError(f'Expected a list as "videos", found {type(videos)}')
        n = len(videos)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]

        import wandb

        metrics = {key: [wandb.Video(video, **kwarg) for video, kwarg in zip(videos, kwarg_list)]}

        for i, video in enumerate(metrics[key]):
            dir = LOG_PATH / key
            dir.mkdir(exist_ok=True, parents=True)
            copy2(video._path, f"{dir}/{step:0>6}_{i}.mp4")

    @rank_zero_only
    def log_table(
        self,
        key: str,
        columns: Optional[list[str]] = None,
        data: Optional[list[list[Any]]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        pass

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = LOG_PATH / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)
