"""HPO-Adapt: diagnostic-driven adaptive HPO setup control.

Core data flow (spec.md):

    TaskSpec -> Setup -> TrialLog -> DiagnosticState -> RLAction -> Setup -> ...
"""

__version__ = "0.1.0"
