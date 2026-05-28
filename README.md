# cno_ratios_analysis
Analysis scripts for ALMA/IRAM CNO ratio analysis

`scripts/fit_iram.py` fits the model to IRAM data.

`condor/fit_iram.sub` runs the model fitting script on each IRAM target. Submit it like:

`condor_submit fit_iram.sub`.

The results are stored in `results/`.

## Outdated
These files have been moved to `old/`.

`scripts/run_bayes_cn_hfs.py` fits the model to ALMA data and compiles the results.

`scripts/analyze_bayes_cn_hfs.py` analyzes the results. Run it like:

`python analyze_bayes_cn_hfs.py {source_name} {data_dir} {results_dir}`

like

`python analyze_bayes_cn_hfs.py G333.052 /path/to/alma_spectra /path/to/results`

It outputs results to a new directory located in `{results_dir}/{source}`.