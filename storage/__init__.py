"""
storage package initializer
Expose a consistent save_quiz / save_learning interface.
Current implementation delegates to local_file_saver; for S3 switch, change these imports.
"""

from .local_file_saver import save_quiz_data as save_quiz
from .local_file_saver import save_learning_data as save_learning

# Backwards compatibility: also expose the module names if other code imports them
# (keep minimal to avoid accidental heavy imports)
# from . import s3_db_saver  # keep commented; switch to real S3 implementation when ready
# from . import storage_interface
