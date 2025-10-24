"""
Compatibility wrapper for pickled objects referencing the legacy module path.

The original data pipeline pickled instances of Node/Relationship from
``local_entity_processing``. After relocating the module under
``innovation_platform.local_entity_processing`` we keep this thin shim so
existing pickle files can still be unpickled without errors.
"""

from innovation_platform.local_entity_processing import *  # noqa: F401,F403
