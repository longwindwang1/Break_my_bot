# Pilots (laptop, 2026-11-09)
Short runs that checked the design and the pipeline before the cloud runs. They are not results, because a full run is 114k iterations; see `../README.md` "Local checks".

| file | content |
|---|---|
| `selection_A.json` | detector grid of the deployed ensemble on the selection splits: diagonal weightings (A) and Mahalanobis weightings (A_cov) |
| `report_A_pipeline_check.json` | A and A_cov on the report splits, from the end-to-end pipeline check |
| `selection_Bshort.json` | B, vicreg, d 128, L 3, 3 000 iterations |
| `selection_P20k.json` | B, same configuration, 20 000 iterations |
| `selection_P20k_wp5.json` | the same with w_pred 5 |
| `probes_P20k.json` | probes on all w4_train envs: A trunk vs P20k z / trunk |
| `diagnose.txt` | encoder sensitivity and residual per event for P20k and P20k_wp5 (`diagnose.py` computes the same) |
