"""HPO-Adapt: diagnostic-driven adaptive HPO setup control.

Core data flow (spec.md):

    TaskSpec -> Setup -> TrialLog -> DiagnosticState -> RLAction -> Setup -> ...

Quick start (wandb-style, see ``hpo_adapt.api``)::

    import hpo_adapt as ha

    run = ha.run(my_train_evaluate, space={...}, direction="maximize",
                 total_trials=120)
    print(run.best_value, run.best_params)
"""

from hpo_adapt.api import Choice, Float, Int, RunResult, run

__version__ = "0.2.0"

__all__ = ["run", "Float", "Int", "Choice", "RunResult", "__version__"]
