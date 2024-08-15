################
 Code Structure
################

In order to ensure sustainable development of the code, when creating a
new feature, the recommended practice is to subclass rather than
re-writing the base classes which would affect other users.

An example of this is shown in
`anemoi/training/diagnostics/callbacks.py` where there is a
`BasePlotCallback` which is called by other plotting callbacks.

If lots of new functions (callbacks for example) are being developed for
a new feature then the recommended practice is to start a new file, for
example `<new_feature>_callbacks.py` to avoid confusion with the base
functions.

Furthermore, always ensure you commit with pre-commit hooks as this
ensure that best practice is followed and never commit directly to
`develop`, instead use a Pull Request from your branch to `develop`.
