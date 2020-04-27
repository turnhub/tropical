#!/usr/bin/env python3
"""
Flask app for local deployment with e.g.:

cd tropical
python flask_app.py --server_port=7100 --prometheus_port=9100
Browse to http://localhost:7100/ui/#/ to see the API.
"""

import os
import logging
import argparse

from tackle.flask_utils import create_flask_app  # noqa
from tackle.flask_utils import setup_logging  # noqa
from tackle.prometheus_utils import create_prometheus_server  # noqa
# from tackle.rest_api import wrapper_util  # noqa
from tackle.rest_api.wrapper_util import add_auth_token  # noqa
from tackle.rest_api import get_path  # noqa


def main():
    server_port = 7100  # Default server port.
    prometheus_port = 9100  # Default server port.
    swagger_ui = True

    ap = argparse.ArgumentParser()
    ap.add_argument("--server_port", required=True, type=int,
                    help=f"The server port. Default is --server_port={server_port}")
    ap.add_argument("--prometheus_port", required=True, type=int,
                    help=f"The server port. Default is --server_port={server_port}")

    args = vars(ap.parse_args())
    server_port = args.get('server_port', server_port)
    prometheus_port = args.get('prometheus_port', prometheus_port)

    print('Arguments =', args)
    print('server_port =', server_port)
    print('prometheus_port =', prometheus_port)

    create_prometheus_server(prometheus_port)

    # Get the logging path from the OS env variable.
    logging_path = os.environ.get("TROPICAL_LOGGING_PATH", "~/.tropical/logs")

    setup_logging(requested_logging_path=logging_path,
                  include_prometheus=True)

    # Get the production or local DB URL from the OS env variable.
    database_url = os.environ.get("TROPICAL_DATABASE_URL",
                                  'sqlite://')

    print("database_url =", database_url)

    # wrapper_util.start_tropical_engine()

    flask_app = create_flask_app(specification_dir=get_path() + '',
                                 add_api=True,
                                 swagger_ui=swagger_ui,
                                 database_url=database_url,
                                 database_create_tables=True if database_url == 'sqlite://' else False,
                                 debug=False)

    # Load some things if needed ...
    # ...

    # === Add some auth tokens to the DB ===
    add_auth_token('tropical-12dd-4104-a7b6-f7d369ff5fec', "Default token for internal hosting.")
    add_auth_token('27deb7c5-8577-451d-b66a-3414e814b353', 'pingdom')
    # === ===

    logging.info(f"rest_flask_app.py: Starting Flask app.")
    flask_app.run(server_port)


if __name__ == '__main__':
    main()
