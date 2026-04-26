# Dataset Citations & Licensing

## The Opportunity Atlas

**Primary citation:**

> Chetty, R., Friedman, J. N., Hendren, N., Jones, M. R., & Porter, S. R. (2020). *The Opportunity Atlas: Mapping the Childhood Roots of Social Mobility* (NBER Working Paper No. 25147). National Bureau of Economic Research. https://doi.org/10.3386/w25147

**Data release:** Publicly available at https://opportunityinsights.org/data/ as `tract_covariates.csv` (tract-level covariates) and outcome files.

**Key description of dataset:** 20.5 million children in the US who were linked to their parents' tax records. The Atlas reports outcomes (income rank, incarceration rate, etc.) at age 35 (for income) or 27 (for incarceration) for these children, aggregated at the census-tract level. The outcomes are reported conditional on parent income quantile (p25 = children from families at the 25th percentile of the national income distribution).

**License:** Public release for research and educational use. See the terms on https://opportunityinsights.org/data/. This project uses the data consistent with its intended research use.

## American Community Survey (ACS) 2015–2019 5-Year Estimates

**Primary citation:**

> U.S. Census Bureau. (2020). *American Community Survey 5-Year Estimates, 2015-2019*. https://www.census.gov/programs-surveys/acs

**License:** Public domain.

## Supporting literature

- Rudin, C. (2019). "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead." *Nature Machine Intelligence*, 1, 206–215.
- Sampson, R. J., Raudenbush, S. W., & Earls, F. (1997). "Neighborhoods and violent crime: A multilevel study of collective efficacy." *Science*, 277(5328), 918–924.
- Wolpert, D. H. (1992). "Stacked generalization." *Neural Networks*, 5(2), 241–259.

## Note on synthetic data

By default, `src/data_loader.py` generates a synthetic tract-level dataset whose marginal distributions and pairwise correlations are calibrated to qualitatively reflect the patterns reported in Chetty et al. (2020) and US Census Bureau ACS tabulations. This synthetic dataset is **not a reproduction of the real Opportunity Atlas data** and should not be used for substantive research claims. It exists to make the pipeline runnable offline and without external dependencies, while still exercising all preprocessing, training, and evaluation steps on a realistic-looking distribution.

The real data is not auto-downloadable as a single file. To adapt this pipeline to real data, you would need to manually obtain:

- **Tract outcomes** from the Census Bureau Opportunity Atlas Data Tables: https://www.census.gov/programs-surveys/ces/data/public-use-data/opportunity-atlas-data-tables.html
- **Tract covariates** from the Harvard Dataverse replication archive: https://doi.org/10.7910/DVN/NKCQM1

Then inspect the actual schema and adapt `src/data_loader.py` accordingly.

This project ships with a calibrated synthetic dataset (see `src/data_loader.py`) rather than the real Atlas. Construction methodology and access points for the real data are documented in `README.md` under *Dataset Construction (Methodology)*.