"""Subcommand modules for the seen18 latent-analysis CLI (see ../seen18.py).

Each module exposes:
  NAME : str                      subcommand name
  HELP : str                      one-line help
  add_args(parser)                register CLI args
  run(args)                       execute the analysis
All share helpers from ../core (io, schemes, metrics, geometry, plot3d).
"""
