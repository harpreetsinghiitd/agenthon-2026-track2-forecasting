# F1 daily full-pipeline forecast charts

Each SVG uses purple for the given historical data and training inputs, blue for the earlier forecast mean, and orange for the later forecast mean. It shows a predictive mean for every forecast date and marks the two forecast cutoffs. At each t1 cutoff, the production F1 local-drift random walk used only panel rows through that cutoff and generated 200 Monte Carlo paths. Text documents were not passed into the forecast; corpus timestamps are grey context markers only. The rolling model selector remains a fallback if the selected drift model fails. Transformation, joint-draw, and LLM conditioning layers are not activated for F1.

- **t2-F1-aud-on-hold-2016** — blue t1 2015-05-01, orange t1 2016-02-03; [chart](t2-F1-aud-on-hold-2016_daily_full_pipeline.svg)
- **t2-F1-cad-boc-2017** — blue t1 2016-01-08, orange t1 2016-10-06; [chart](t2-F1-cad-boc-2017_daily_full_pipeline.svg)
- **t2-F1-eur-range-2017** — blue t1 2016-03-08, orange t1 2016-12-06; [chart](t2-F1-eur-range-2017_daily_full_pipeline.svg)
- **t2-F1-jpy-ycc-flex-2018** — blue t1 2017-01-30, orange t1 2017-10-27; [chart](t2-F1-jpy-ycc-flex-2018_daily_full_pipeline.svg)
