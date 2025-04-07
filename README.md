# cno_ratios_analysis
Analysis scripts for ALMA/IRAM CNO ratio analysis

`scripts/run_bayes_cn_hfs.py` fits the model to ALMA data and compiles the results.

`scripts/analyze_bayes_cn_hfs.py` analyzes the results. Run it like:

`python analyze_bayes_cn_hfs.py {source_name} {data_dir} {results_dir}`

like

`python analyze_bayes_cn_hfs.py G333.052 /path/to/alma_spectra /path/to/results`

It outputs results to a new directory located in `{results_dir}/{source}`.