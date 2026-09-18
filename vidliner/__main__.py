"""Support ``python -m vidliner`` as an alias for the ``vidliner`` console script.

Both routes reach the same Typer application, so behaviour and exit codes are identical whether the
package was installed with its script or is being run straight from a checkout.
"""

from __future__ import annotations

from vidliner.cli.main import app

__all__ = ["app"]


def main() -> None:
    """Run the command line application."""
    app()


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
