# 1.0.4 🇺🇦 🇺🇦 🇺🇦
* more tuning of MCP download install
* TVA optimizations
* Synthetic data minor new options
* Feature detector and associated transformer bug fixes

### FastARIMA / FastProphet follow-ups (added in branch feat/fast-arima-prophet-tvvar)
* [feat/fast-arima-prophet-tvvar] AGENTS.md alignment: rename short variables (u, w, p, q, t, c, sl, y_c, m_c) to descriptive names
* [feat/fast-arima-prophet-tvvar] AGENTS.md alignment: trim extra fallbacks (per-series lstsq, zeroing non-finite coefs, AR-only short-history path) in favor of failing so the search tries other params
* [feat/fast-arima-prophet-tvvar] run a formatter (black-style, project has none configured) on the new files; ~37 lines are > 88 chars
* [feat/fast-arima-prophet-tvvar] decouple FastProphet.get_new_params from FBProphet (copy the search space)
* [feat/fast-arima-prophet-tvvar] finish New Model Checklist: extended_tutorial.md model table, docs, decide on all_models / templates
* [feat/fast-arima-prophet-tvvar] FastARIMA: process series in chunks (like FastProphet's element budget); peak RAM grows ~0.17-0.29 MiB/series (1.45 GiB at 5000 series, ARMA(5,5), 1000 rows)
* [feat/fast-arima-prophet-tvvar] measure peak memory of ARIMA/FBProphet with n_jobs > 1 (only n_jobs=1 was measured)
* [feat/fast-arima-prophet-tvvar] FastTVVAR: decide whether to fix TVVAR's var_dampening slice ([-n_var:-1] skips the last VAR column; with no lags it dampens exogenous columns). FastTVVAR dampens the full VAR block
* [feat/fast-arima-prophet-tvvar] FastTVVAR: public process_components() is still TVVAR's pandas version (PredictionObject.components is vectorised)
* [feat/fast-arima-prophet-tvvar] TVVAR bug: var_postprocessing result is discarded (`inverse_transform(df)` not assigned), so it never changes forecasts; FastTVVAR skips it for parity. Decide whether to fix (would change TVVAR results) or remove the param
* [feat/fast-arima-prophet-tvvar] TVVAR bug: components are silently None when base_scaled=False (indexes float scaler_mean by series); FastTVVAR returns them
* [feat/fast-arima-prophet-tvvar] FastTVVAR: New Model Checklist (extended_tutorial.md table, model lists beyond model_classes, mcp explainable model list)

### New Model Checklist:
* Add to ModelMonster in auto_model.py
* add to appropriate model_lists: all, recombination_approved if so, no_shared if so
* add to model table in extended_tutorial.md (most columns here have an equivalent model_list)
* if model has regressors, make sure it meets Simulation Forecasting needs (method=="regressor", fails on no regressor if "User")
* if model has result_windows, add to appropriate model_list noting also diff_window_motif_list

## New Transformer Checklist:
* Make sure that if it modifies the size (more/fewer columns or rows) it returns pd.DataFrame with proper index/columns
* add to transformer_dict
* add to trans_dict or have_params or external as appropriate
* add to shared_trans if transformer is multivariate (would produce different results if run on each series individually than if run on all together)
* add to oddities_list for those with forecast/original transform difference
* add to docstring of GeneralTransformer
* add to dictionary by type: filter, scaler, transformer
* add to test_transform call

## New Metric Checklist:
* Create function in metrics.py
* Add to mode base full_metric_evaluation  (benchmark to make sure it is still fast)
* Add to concat in TemplateWizard (if per_series metrics will be used)
* Add to concat in TemplateEvalObject (if per_series metrics will be used)
* Add to generate_score
* Add to generate_score_per_series (if per_series metrics will be used)
* Add to validation_aggregation
* Update test_metrics results
* metric_weighting in AutoTS, get_new_params, prod example, test
