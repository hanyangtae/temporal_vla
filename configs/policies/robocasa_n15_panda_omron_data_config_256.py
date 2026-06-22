"""256x256 variant of RobocasaPandaOmron10TaskDataConfig (B3 resolution probe).

Identical to ``RobocasaPandaOmron10TaskDataConfig`` EXCEPT the ``VideoResize``
target is 256x256 instead of 224x224. Used to test whether feeding 256-px frames
(as COAST does) raises the N1.5 faithful-native baseline SR toward COAST's 0.59.

Subclasses and mutates the ``VideoResize`` instance returned by the parent
``transform()`` so that ``set_metadata`` builds the resize at 256.
"""

import os

from robocasa_n15_panda_omron_data_config import RobocasaPandaOmron10TaskDataConfig


class RobocasaPandaOmron10TaskDataConfig256(RobocasaPandaOmron10TaskDataConfig):
    def transform(self):
        t = super().transform()
        for tr in getattr(t, "transforms", []):
            if type(tr).__name__ == "VideoResize":
                tr.height = 256
                tr.width = 256
        # Verification: surface the actual VideoResize target so the serve log
        # proves 256 is applied (gated behind env var to stay quiet by default).
        if os.environ.get("B3_VERIFY_RESIZE"):
            for tr in getattr(t, "transforms", []):
                if type(tr).__name__ == "VideoResize":
                    print(
                        f"[B3-VERIFY] VideoResize height={tr.height} width={tr.width}",
                        flush=True,
                    )
        return t
