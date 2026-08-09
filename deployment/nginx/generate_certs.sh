#!/bin/sh
# Generates a self-signed certificate in ./certs folder (development only)
set -e
mkdir -p certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout certs/server.key -out certs/server.crt -subj "/C=US/ST=State/L=City/O=Example/CN=localhost"
echo "Generated certs/server.key and certs/server.crt"
