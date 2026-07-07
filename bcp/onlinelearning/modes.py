from dataclasses import dataclass


@dataclass(frozen=True)
class OnlineLearningMode:
    closedloop: bool = False
    update_wFF: bool = False
    update_wOUT: bool = False
    update_wEE: bool = False
    update_wIE: bool = False

    @classmethod
    def open_loop_eval(cls):
        return cls(closedloop=False)

    @classmethod
    def closed_loop_eval(cls):
        return cls(closedloop=True)

    @classmethod
    def train_full(cls, update_wEE=False):
        return cls(
            closedloop=True,
            update_wFF=True,
            update_wOUT=True,
            update_wEE=update_wEE,
            update_wIE=False,
        )

    @classmethod
    def train_readout(cls):
        return cls(
            closedloop=False,
            update_wFF=False,
            update_wOUT=True,
            update_wEE=False,
            update_wIE=False,
        )

    @classmethod
    def train_wie_only(cls):
        return cls(
            closedloop=False,
            update_wFF=False,
            update_wOUT=False,
            update_wEE=False,
            update_wIE=True,
        )
