"""
WSGI Flask App for production hosting with e.g.:

gunicorn --bind 0.0.0.0:80 -w 1 -t 120 wsgi
"""

import os
import sys
import logging

# NOTE: The below import is useful to bring feersum_nlu into the Python path!
module_path = os.path.abspath(os.path.join('..'))
print("module_path =", module_path, flush=True)
if module_path not in sys.path:
    sys.path.append(module_path)

from tackle.flask_utils import create_flask_app  # noqa
from tackle.flask_utils import setup_logging  # noqa
from tackle.prometheus_utils import create_prometheus_server  # noqa
# from tackle.rest_api import wrapper_util  # noqa
from tackle.rest_api.wrapper_util import add_auth_token  # noqa
from tropical.rest_api import get_path  # noqa

create_prometheus_server(9101)

logging_path = os.environ.get("TROPICAL_LOGGING_PATH", "~/.tropical/logs")
setup_logging(requested_logging_path=logging_path,
              include_prometheus=True)

# Get the production or local DB URL from the OS env variable.
database_url = os.environ.get("TROPICAL_DATABASE_URL", 'sqlite://')

# wrapper_util.start_tropical_engine()

flask_app = create_flask_app(specification_dir=get_path() + '',
                             add_api=True,
                             swagger_ui=True,
                             database_url=database_url,
                             database_create_tables=True if database_url == 'sqlite://' else False,
                             debug=False)

# Load some things if needed...
# ...

# === Add some auth tokens to the DB ===
add_auth_token('tropical-12dd-4104-a7b6-f7d369ff5fec', "Default token for internal hosting.")
add_auth_token('27deb7c5-8577-451d-b66a-3414e814b353', 'pingdom')
# === ===

logging.info(f"rest_wsgi_app.py: __name__ == {__name__}")
application = flask_app.app

if __name__ == "__main__":
    logging.info(f"rest_wsgi_app.py: __main__ Starting Flask app in Python __main__ .")
    flask_app.run()
