# Data and pipeline provenance — draft appendix
*Protecting Consumers from Credit Card Fraud* · V01, 2026-10-12 · Revision_Plan_V06.md Stages 2B and 2D

This is the evidence for a short "Data and pipeline provenance" appendix and for the rewritten §3.1
and §3.3. Every number in it comes from a committed script, cited in the text.

---

## 1. The transaction data and the generator

**Dataset.** The data are the Kaggle *Credit Card Transactions Fraud Detection Dataset* (user
`kartik2112`), distributed as `fraudTrain.csv` and `fraudTest.csv`.

| | fraudTrain | fraudTest | Both |
|---|---|---|---|
| Period | 2019-01-01 – 2020-06-21 | 2020-06-21 – 2020-12-31 | |
| Rows | 1,296,675 | 555,719 | |
| Fraudulent | 7,506 | 2,145 | |
| Cards | 983 | 924 | 999 |
| Merchants | 693 | | 693 |

There are 999 cardholders across both files: each card belongs to a distinct name and date of birth.

**The manuscript's "1,000 customers and a pool of 800 merchants" does not match the data.** The
data hold 999 cards and 693 merchants, and fraudTrain alone holds 983 cards. §3.1 should give the
counts observed in the file the study uses.

**Generator.** The dataset was produced with *Sparkov Data Generation* (Brandon Harris,
github.com/namebrandon/Sparkov_Data_Generation). Its README states that "the generation code is
originally based on code by Josh Plotkin" (github.com/joshplotkin/data_generation).

- **What the generator does.**
  - It is written in Python and uses `faker` for customer identities.
  - It draws customers from demographic profiles defined in JSON configuration files.
  - Transactions are generated per profile.
  - Transaction amounts are drawn from Gamma distributions.
  - Merchants are drawn per spending category, with locations randomised around the customer's
    home and a travel probability.
  - Fraud enters through *fraudulent profiles* and *fraudulent transactions* (README changelog,
    v0.3).
- **Which version produced this dataset.**
  - The repository's history has a merged pull request from `kartik2112` — the Kaggle uploader —
    in July 2020 ("Bat files and README updated"). The next commits are from June 2022: a refactor
    and multiprocessing parallelisation, released as v0.5/v1.0.
  - The dataset therefore comes from the **mid-2020 state of the code** (README changelog v0.1–v0.4,
    including the v0.3 fraud profiles and Gamma amounts), not from the current version.
  - `kartik2112`'s own fork lists modifications up to v0.4.
  - **The authors should confirm this against the Kaggle page (J3.1).**
- **What it does not do.** Neither the Sparkov README nor Plotkin's original ("pure python, HAWQ
  (with PL/Python), or MapReduce (streaming via python)") mentions Apache Spark or Markov chains.
  The name reads as a portmanteau, not as a description of the method.
  - **§3.1's account ("combined two techniques: (1) Apache Spark … and (2) Markov Chains") should
    be removed**, together with references [43] and [44] if nothing else cites them.

**Proposed §3.1 wording** (for the rewrite):

> The data were generated with Sparkov Data Generation, an open-source Python tool that simulates
> card transactions from demographic customer profiles: identities are drawn with the `faker`
> library, transaction amounts from per-profile Gamma distributions, and merchants per spending
> category around each customer's home location. Fraud is introduced through designated fraudulent
> profiles and transactions, so which transactions are fraudulent is fixed by the generator's
> configuration rather than by any offender behaviour. We use the publicly distributed dataset
> produced with the mid-2020 version of the tool [Kaggle ref; GitHub ref]. Its training file holds
> 1,296,675 transactions from 983 cards and 693 merchants between 1 January 2019 and 21 June 2020,
> of which 7,506 (0.58%) are fraudulent.

## 2. External reference data — observation period and release

All files are under `Python_proj_code/MLProj01_US/encoding_table/ref_data/original/`. The
transaction window is 2019-01 to 2020-06 (fraudTrain), extended to 2020-12 by fraudTest.

| Feature | Source and file | Observation period | Release / version | Variable extracted | After the window? |
|---|---|---|---|---|---|
| `city_1` county median household income quartile | USDA ERS, *Unemployment and median household income*, `Unemployment_Income_2023.csv` | 2022 | 2023 release | `Median_Household_Income_2022` (the file's only income year) | **yes** |
| `city_2` county Gini quintile | US Census ACS 5-year, table B19083, `ACSDT5Y2020.B19083-Data.csv` | 2016–2020 | 2020 5-year | `B19083_001E` | ends Dec 2020 (overlaps) |
| `city_3` urban type (rural/suburban/urban) | USDA ERS *Urban Influence Codes*, `UrbanInfluencecodes2024.csv` | 2020 census / 2024 delineation | 2024 | `UIC_2024`, mapped 1,4 → urban; 2,3,5,7 → suburban; 6,8,9 → rural | **yes** |
| `city_4a` fixed broadband quintile | FCC Broadband Data Collection, `bdc_us_fixed_…_D22_04nov2024.csv` | as of Dec 2022 | release of 4 Nov 2024 | mean coverage at 100/20, 250/25 and 1000/100 Mbps, county totals | **yes** |
| `city_4b` 5G coverage quintile | FCC BDC mobile, `bdc_us_mobile_…_D22_04nov2024.csv` | as of Dec 2022 | 4 Nov 2024 | mean of 5G speed-1 and speed-2 area coverage | **yes** |
| city → county | ZIP code database, `zip_code_database.csv` (unitedstateszipcodes.org layout) | undated, current | downloaded ~Apr 2025 | (`state`, `primary_city`) → `county` | n/a (geography) |
| state names | SimpleMaps, `uscities.csv` | — | downloaded Jan 2025 | state name ↔ code only | n/a |
| `job_1` occupation income quartile | BLS OEWS national, `national_M2020_dl.xlsx` (via `ref_data/job_income.csv`) | May 2020 | 2021 | `A_MEAN` | **no** (inside the window) |
| `job_2` required education | O\*NET *Education, Training, and Experience* | ratings dated 2004–2024 | O\*NET ~29.x (2024) | modal "Required Level of Education" category, binned ≤1.5 / ≤5.5 / above | **mixed** |
| `job_3` career level | O\*NET *Job Zones* | as above | as above | Job Zone − 1 | **mixed** |
| `job_4` "cyber knowledge" | O\*NET *Knowledge*, element "Computers and Electronics", scale "Level" | as above | as above | level, quintiles over all O\*NET occupations | **mixed** |

**Manual fills** (encoding cells 16–21), for counties the reference files do not resolve:
- 5 cities' counties added by hand.
- Income for 8 counties (data.census.gov profiles, 2022 values).
- Gini for 4 (datausa.io, "in 2022"). The three Virginia independent cities share one value, 0.475,
  and Norfolk's has no source given.
- Urban type for 9 (Connecticut planning regions, IL LaSalle, MO St Louis, VA independent cities;
  regional documents).
- Fixed broadband for 4 and 5G for 4 (broadbandnow.com and bestneighborhood.org, current web pages).

**Assessment.** Most enrichment postdates the transactions. A system scoring transactions in
2019–20 could not have used the 2022–24 values. The paper therefore labels the county and occupation
attributes as **retrospective contextual enrichment**, describing the place or occupation rather
than information available at prediction time. It reports the configuration that excludes them
(`p7_abl_no_external`: no `city_1..4b` and no `job_1..4`). Phase 6 found these features carry no
signal in this dataset (§8C), to be re-measured on the corrected data, so the exclusion is expected
to cost nothing. If so, the vintage question cannot affect any reported result. (Decision of
2026-10-12: label and exclude, rather than rebuild from 2019–20 vintages.)

**Proxies, not measurements.**
- Every county attribute describes the cardholder's *county of residence* as resolved from the
  city name, not the cardholder.
- Every occupation attribute describes the *O\*NET occupation nearest to the Sparkov job title in
  sentence-embedding space*. The match has no similarity threshold, so every title gets a nearest
  match. It does not describe the person.
- "Cyber knowledge" in particular is the occupation's rated need for knowledge of computers and
  electronics. It is not anyone's cybersecurity knowledge. The paper should call it "occupational
  computer knowledge (O\*NET rating)".

**Resolution choices to state.**
- Of the 923 cities matched through the ZIP database, 31 lie in more than one county. The pipeline
  takes the alphabetically first county.
- The city lookup is now asserted to hold one row per (state, city), in encoding_V25.

## 3. Learned preprocessing — where it was fitted

`revision_scripts/fit_scope_check.py` → `result_csv/fit_scope_check_V01.csv`, with
`category_cost_check.py` → `result_csv/category_cost_check_V01.csv`. No step uses a label.

| Step | Fitted on | Training split alone gives | Effect |
|---|---|---|---|
| `category_3` cost quartiles (14 category means) | all rows | the identical map | none |
| `job_1` income quartiles (494 titles, unweighted) | all titles in the file | the identical quartile for every title (490 of 494 occur in training) | none |
| location grid bounds (`loc_cluster_issue/transac`) | min/max lat/long over all rows | the same bounds except maximum longitude (−66.9509 vs −66.9565) | 5,614 cardholder cells and 1,749 merchant cells move (0.43% / 0.13% of rows; 2,095 and 630 after training). Label-free, but a full-sample statistic: §3.3 states it |
| county income, Gini, broadband, 5G quantiles | all US counties (external) | — | not data-derived |
| `job_4` quintiles | all O\*NET occupations (external) | — | not data-derived |
| education, career level | fixed bins | — | — |
| raw baseline `city_ordinal`, `job_ordinal` | alphabetical rank over all levels (no statistic) | — | 22 cities and 4 jobs first occur after training: 211 and 41 transactions, **all fraudulent** (the fraud-only cardholders) |
| logistic-regression scaler | training split | — | — |
| LIME explainer statistics | training split | — | — |
| hyperparameters, early stopping, thresholds | validation split | — | — |

The unseen raw levels deserve a sentence in §3.3. A city first seen after training gets an
arbitrary ordinal code under R, but its true county attributes under E. Every such transaction is
fraudulent, so the two representations handle the new-cardholder population differently by
construction.

## 4. Client-level behavioural features — conventions

`revision_scripts/client_features.py`, verified against `fraudTrain_encoded_client.csv` (integer
columns exact, floats to 1e-13). Every feature uses only the same card's **strictly earlier**
transactions and never `is_fraud`. A brute-force check over 25 random cards passes
(`MLProj01_fraudTrain_clientfeatures_V01.ipynb` cell 6).

| Convention | Rule |
|---|---|
| Ordering | by card, then `unix_time`, then original row (ties keep file order) |
| Timestamp ties | `card_hours_since_prev` = 0; a tied earlier-listed transaction counts as prior, a later-listed one does not |
| Cold start | `card_amt_zscore` = 0 until 5 prior transactions; `card_hours_since_prev` = 720 h (also the cap) for the first; trailing counts = 0; `card_dist_from_usual_km` measured from the cardholder's home for the first; `card_new_category` = 1 |
| Zero variance | z-score = 0 when the prior amounts have zero spread; clipped to ±10; sample variance (ddof = 1) |
| History through validation and test | yes. History runs over the whole series, so a validation or test transaction's features use earlier validation or test transactions of the same card — causal and label-free, as in online scoring |
| Left censoring | every card starts without history on 2019-01-01 |
| fraudTest | history continues from fraudTrain (the two series concatenated in time) |
| Fraud-only cards | on the 75 cards that carry only fraud, "history" is fraudulent history |
