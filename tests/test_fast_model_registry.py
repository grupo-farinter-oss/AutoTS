"""Fast* vectorised models must be registered wherever their original is.

Why this matters: ``recombination_approved`` gates genetic crossover of parameters in
NewGeneticTemplate, and ``all_models`` is what consumers read to learn which models exist.
A Fast model missing from either silently degrades the search or gets rejected.
"""

import unittest

from autots.models.model_list import all_models, model_classes, model_lists

# Fast model -> the model it replaces
FAST_TO_ORIGINAL = {
    "FastTVVAR": "TVVAR",
    "FastARIMA": "ARIMA",
    "FastProphet": "FBProphet",
}

# Lists where a Fast model is intentionally NOT registered although its original is.
# Reasons are checked against the code, see PR description.
INTENTIONAL_EXCEPTIONS = {
    "slow": "Fast models are the fast version; 'slow' means slower than the fast list",
    "parallel": "FastARIMA/FastProphet accept n_jobs but never use it (no joblib); "
    "'parallel' is for models that scale with n_jobs",
    "default": "default holds curated weights for the original models; a Fast twin "
    "would duplicate TVVAR's candidates (identical forecasts) in the default search",
}


class FastModelRegistryTests(unittest.TestCase):
    def test_fast_models_registered_wherever_original_is(self):
        for fast_name, original in FAST_TO_ORIGINAL.items():
            for list_name, members in model_lists.items():
                if list_name in INTENTIONAL_EXCEPTIONS or original not in members:
                    continue
                with self.subTest(model=fast_name, list=list_name):
                    self.assertIn(
                        fast_name,
                        members,
                        f"{fast_name} missing from '{list_name}' which contains {original}",
                    )

    def test_intentional_exceptions_hold(self):
        for fast_name in FAST_TO_ORIGINAL:
            for list_name in INTENTIONAL_EXCEPTIONS:
                with self.subTest(model=fast_name, list=list_name):
                    self.assertNotIn(fast_name, model_lists[list_name])

    def test_fast_models_in_all_models(self):
        for fast_name in FAST_TO_ORIGINAL:
            with self.subTest(model=fast_name):
                self.assertIn(fast_name, all_models)
                self.assertIn(fast_name, model_lists["all"])
                self.assertIn(fast_name, model_lists["all_pragmatic"])

    def test_every_fast_class_is_declared_here(self):
        """A new Fast* model must be paired with its original so this test covers it."""
        declared = set(FAST_TO_ORIGINAL)
        for name in model_classes:
            if name.startswith("Fast"):
                with self.subTest(model=name):
                    self.assertIn(
                        name,
                        declared,
                        f"{name} is in model_classes: add it to FAST_TO_ORIGINAL "
                        "and register it in model_list.py",
                    )
        for fast_name, original in FAST_TO_ORIGINAL.items():
            self.assertIn(original, all_models)
            self.assertIn(fast_name, model_classes)

    def test_registration_matches_model_behaviour(self):
        """Claims made by list membership that can be checked directly."""
        import inspect
        from autots.models import fast_arima, fast_prophet

        # not in 'parallel': n_jobs is passed through but never drives parallel work
        for module in (fast_arima, fast_prophet):
            self.assertNotIn("joblib", inspect.getsource(module))
            self.assertNotIn("Parallel(", inspect.getsource(module))
        # horizontal-subset logic relies on no_shared meaning "independent per series"
        for name in ("FastARIMA", "FastProphet"):
            self.assertIn(name, model_lists["no_shared"])
            self.assertIn(name, model_lists["univariate"])
            self.assertNotIn(name, model_lists["multivariate"])
        self.assertIn("FastTVVAR", model_lists["multivariate"])
        self.assertNotIn("FastTVVAR", model_lists["no_shared"])
        self.assertNotIn("FastTVVAR", model_lists["univariate"])


if __name__ == "__main__":
    unittest.main()
