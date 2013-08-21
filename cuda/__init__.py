"""Some cuda utility functions.

Note for unittest: to make sure a cuda test is carried out only when there
are cuda libraries on the local machine, we define a flag has_cuda to indicate
whether tests could be skipped.
"""

import logging

try:
    from cudawrapper import *
    has_cuda = True
except OSError:
    logging.error('No working cuda library found. Any call to cuda functions '
                  'will fail.')
    has_cuda = False
