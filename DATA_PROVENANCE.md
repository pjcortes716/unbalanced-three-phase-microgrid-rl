# Data provenance

This snapshot distributes exactly eight derived NPZ pools: four `train` and four `optuna_validation`. Each contains `weeks` (float64, n x 168) and `base_ids` (text, no pickled objects). Full raw data, original unpartitioned pools, and physical `final_test` datasets are absent. The manifest retains holdout IDs, dates, source lineage and hashes solely to document the partition.

## Building loads and source attribution

The dated loads come from the local Building Data Genome 2 (BDG2) v1.0 archive, directory revision suffix `3d0cbaf`, specifically `data/meters/cleaned/electricity_cleaned.csv`. The suffix identifies the local archive, not an independently verified full Git revision. The distributed manifest's historical `source_file` fields are provenance labels; the environment does not resolve or open them.

| Role | Original column |
| --- | --- |
| Office | `Panther_office_Karla` |
| Warehouse 1 | `Hog_warehouse_Louise` |
| Warehouse 2 | `Rat_warehouse_Shari` |
| Calendar | `timestamp` |

The source covers 2016-01-01 00:00 through 2017-12-31 23:00 (17,544 hours). Timestamps are naive: no physical timezone is assigned. Individual buildings and weeks are used as profiles in a simulated microgrid, not evidence of one co-located measured industrial site. Original hourly values are retained in NPZ; the existing samplers later apply peak scaling, stochastic realization and synthetic forecast noise.

Attribution retained from the local BDG2 README: Clayton Miller, Anjukan Kathirgamanathan, Bianca Picchetti, Pandarasamy Arjunan, June Young Park, Zoltan Nagy, Paul Raftery, Brodie W. Hobson, Zixiao Shi, and Forrest Meggers (2020), *The Building Data Genome Project 2: Hourly energy meter data from the ASHRAE Great Energy Predictor III competition*. [Source project](https://github.com/buds-lab/building-data-genome-project-2); [paper recorded by the local archive](https://arxiv.org/abs/2006.02273).

## Monday-Sunday reconstruction and filters

The approved builder sorts timestamps, rejects duplicates, groups using `W-SUN`, and requires exactly 168 unique consecutive hourly timestamps from Monday 00:00 to Sunday 23:00. It requires finite, nonnegative values and does not interpolate or impute missing measurements.

Office coverage starts on 2016-05-23 to preserve the historical extractor's accepted coverage. Warehouse filters allow at most 15% zeros and require weekly sum >=20 in the original series units. All filter parameters and per-group discard reasons are preserved in the manifest.

| Source | Examined | Valid | Discarded |
| --- | ---: | ---: | ---: |
| Office | 105 | 81 | 24 |
| Warehouse 1 | 105 | 67 | 38 |
| Warehouse 2 | 105 | 97 | 8 |
| Solar | 80 | 80 | 0 |

Discard-reason counts may overlap. Office has one incomplete group, 21 groups outside the accepted start, and three nonfinite groups; warehouse 1 has one incomplete and 37 nonfinite; warehouse 2 has one incomplete and seven nonfinite. No additional groups were removed solely for the warehouse zero/sum thresholds.

## Solar provenance limitation

The original local `solar_weeks_pool.npz` contains 80 generic profiles with 168 hours. There is no verified extraction procedure linking those rows to an original meter column, location or calendar dates. A nearby `solar_cleaned.csv` is not sufficient evidence to attribute these rows to that file. Solar profiles remain generic with `start=null`, `end=null`, `original_index`, base ID and hash. The name of the containing BDG2 directory does not prove that BDG2 owns or licenses these solar profiles.

## Split protocol

Separation precedes noise, ABC allocation, prices, SOC and forecasts. The 54 common valid Mondays across the three dated sources determine the holdouts: the last 12 common Mondays are split into the earlier six for validation and later six for final testing. Solar reserves the last 12 valid original indices as 6+6 without claiming chronology. All other valid profiles enter train: 69 office, 55 warehouse 1, 85 warehouse 2, 68 solar. Each evaluation split has six profiles per source and six fixed reference compositions.

Train independently combines sources. Some non-common train weeks are later than holdout dates: this is base-profile isolation, not a strictly prospective temporal experiment. Seeds do not move base profiles between splits. The catalog validates disjoint/exhaustive IDs, profile hashes and dated weeks, including cross-source date isolation. Selection uses no reward or policy results. `final_test` must not be used for hyperparameters, coefficient selection, pruning or checkpoint selection.

`data/final_test` does not exist. Its four expected NPZ hashes remain metadata and were not reverified against physical holdout files during packaging. No final-test rollout or series access was needed.

## Hashes and structural relocation

Only the twelve relative NPZ path strings in the manifest changed from `benchmark_v1_data/<split>/...` to `<split>/...`, because the manifest now lives in `data/`. All other manifest content, profile hashes and NPZ hashes are unchanged. This changes the manifest hash and canonical scenario-ID prefix; use IDs generated by this snapshot rather than previously recorded IDs.

- Original manifest SHA-256: `0db4a1ccbceec4833d6c94a131efe9e5060b6e09b43198522563269ff2c0f542`.
- Snapshot manifest SHA-256: `91b28e5d82d79da488f167eb7cd3d9dcb9ee805b3726dc232b63535e62c91830`.
- Original electricity CSV SHA-256 (metadata): `b01056f4ceec1992ce019297e674d6d6ecd0240e50a2aaeb6f958c65efdc282d`.
- Original solar pool SHA-256 (metadata): `161c01621b0de8413e1e0a8c2636c757a785457750da0f9fb60761d4fef198df`.

| Included pool | Weeks | SHA-256 (byte-preserved) |
| --- | ---: | --- |
| `train/office` | 69 | `d1e9a8938b24ca06e102eb5a2b1102d95909214ffdec51f7c86a463d8df2077c` |
| `train/wh1` | 55 | `052b15879177b3241ca1af23f86fa5d50bd962bac12af727458ee8292f11b6b7` |
| `train/wh2` | 85 | `2af32b0cdfe818f55086a57fa56cd7ba559c2b8beace8147e922cfa57d698ff3` |
| `train/solar` | 68 | `07cde2e9d29184583231c8c02ec262d0cf331418c64e3e0ad805c4410cbf27c8` |
| `optuna_validation/office` | 6 | `a4c943b93d194978d8e75badf150f616c5dc67d74706913e3159f40f3103db62` |
| `optuna_validation/wh1` | 6 | `a8544196006546855a0d05aeb31e14e7e289ecb26b69c2684cb1e903fcca2963` |
| `optuna_validation/wh2` | 6 | `1cc3c7c56d54ff94352b7b4e82b06564371b88cf8bcaab978a273d60787fd84b` |
| `optuna_validation/solar` | 6 | `c7579f37e4715156400704760eeb4e53f1ed2f92a5b4594fbb08f13aaf9b17e6` |

Check the distributed hashes with:

```bash
python -B scripts/prepare_benchmark_v1_data.py --verify-only
```

The copied builder retains the approved extraction rules. Full rebuilding requires independently supplied originals matching the recorded hashes and a **new** output directory:

```bash
python -B scripts/prepare_benchmark_v1_data.py --electricity-csv ORIGINAL_CSV --solar-npz ORIGINAL_SOLAR_NPZ --output-dir NEW_OUTPUT_DIRECTORY
```

The distributed builder never writes final_test arrays. No rebuilding or raw-source verification was run during this preparation. The two adapted nominal checks validate included values against recorded hashes and reconstruct timestamps from metadata; they do not re-establish correctness against the complete original CSV or original solar pool.

## Licenses and open provenance items

The supplied local BDG2 archive has an MIT license notice, copyright 2020 Building and Urban Data Science (BUDS) Group; it is reproduced below. The current upstream [LICENSE](https://github.com/buds-lab/building-data-genome-project-2/blob/master/LICENSE), inspected 2026-09-09, instead carries Attribution-ShareAlike wording. These differing notices require reconciling the applicable release and the terms covering the actual electricity data before public redistribution. The local notice alone is not a blanket MIT grant for all external datasets. Neither notice establishes a license for the untraced solar pool.

The root PolyForm Noncommercial License 1.0.0 applies to original software code only. Third-party and derived datasets remain subject to the terms, licenses, and attribution requirements of their respective original sources. Neither PolyForm Noncommercial nor the CC BY-NC-SA 4.0 license for original documentation automatically applies to datasets. Original documentation prose in this file is covered by CC BY-NC-SA 4.0 as stated in README.md; quoted upstream license text and attribution notices retain their own terms. Before public release: resolve the dated-load license/version discrepancy, establish solar authorship and redistribution permission or replace the profiles with documented data, verify attribution for the electrical parameter table/paper referenced by the audit, and confirm the code contributors' rights. The private snapshot preserves provenance gaps rather than inventing sources. Also document the synthetic price/forecast assumptions and limitations when writing the article.

### License notice found in the local BDG2 archive

The following third-party notice is retained as evidence and attribution for the local BDG2 archive only. It does not license the original software in this repository and is not replaced by PolyForm or the documentation license:

```text
MIT License

Copyright (c) 2020 Building and Urban Data Science (BUDS) Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
