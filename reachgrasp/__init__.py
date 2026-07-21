"""Python utilities for the ReachGrasp dataset.

The package initializer is intentionally lightweight. Import functionality from
its owning module, for example::

    from reachgrasp.data_loading import load_joint_angles_with_labels
    from reachgrasp.preprocessing import bandpass_filter

Keeping the initializer free of eager re-exports means that data-loading code
does not also import plotting libraries or unrelated analysis modules.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
