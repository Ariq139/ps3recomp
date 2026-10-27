"""The application has no model backend, downloads, or online services."""
import os
import sys


def enforce_offline():
    os.environ["PIP_NO_INDEX"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    def audit(event, args):
        if event in ("socket.connect", "socket.connect_ex", "socket.sendto"):
            raise PermissionError("Offline reconstruction does not permit Python network connections")
    sys.addaudithook(audit)
