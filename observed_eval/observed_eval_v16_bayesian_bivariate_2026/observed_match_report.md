# v16_bayesian_bivariate_model observed World Cup evaluation

- Matches: 12
- Result accuracy: 50.0% (95% CI 25.0% to 75.0%)
- Result log loss: 1.017 (95% CI 0.745 to 1.283)
- Three-way Brier score: 0.625 (95% CI 0.426 to 0.819)
- Ranked probability score: 0.198 (95% CI 0.132 to 0.272)
- Goal MAE: 0.904 (95% CI 0.612 to 1.234)
- Goal RMSE: 1.356 (95% CI 0.794 to 1.812)
- Exact-score accuracy: 8.3%
- Log-loss skill vs uniform forecast: 7.5%

Bootstrap intervals measure sampling uncertainty across this observed match set. With a small number of matches, they should be expected to be wide.

## Subgroups

| dimension | subgroup | n_matches | result_accuracy | mean_result_log_loss | mean_result_brier | mean_result_rps | mean_goal_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 12 | 0.500 | 1.017 | 0.625 | 0.198 | 0.904 |
| actual_result | draw | 4 | 0.000 | 1.426 | 0.952 | 0.188 | 0.554 |
| actual_result | team_a_win | 7 | 0.714 | 0.851 | 0.490 | 0.217 | 1.120 |
| actual_result | team_b_win | 1 | 1.000 | 0.534 | 0.258 | 0.103 | 0.795 |
| stage | Group Stage | 12 | 0.500 | 1.017 | 0.625 | 0.198 | 0.904 |
| group | Group A | 2 | 1.000 | 0.526 | 0.261 | 0.103 | 0.321 |
| group | Group B | 2 | 0.000 | 1.514 | 1.044 | 0.218 | 0.575 |
| group | Group C | 2 | 0.500 | 0.903 | 0.544 | 0.129 | 0.571 |
| group | Group D | 2 | 0.500 | 0.980 | 0.588 | 0.257 | 1.178 |
| group | Group E | 2 | 0.500 | 1.014 | 0.594 | 0.279 | 1.477 |
| group | Group F | 2 | 0.500 | 1.162 | 0.718 | 0.200 | 1.305 |

## Box events

| event | n_team_observations | mae | rmse | bias | correlation |
| --- | --- | --- | --- | --- | --- |
| shots | 24 | 5.161 | 6.309 | 1.028 | 0.393 |
| shots_on_target | 24 | 1.948 | 2.283 | 0.694 | 0.563 |
| possession | 24 | 7.026 | 8.323 | 0.000 | 0.771 |
| fouls | 24 | 5.054 | 5.860 | 3.175 | 0.107 |
| yellow_cards | 24 | 1.139 | 1.295 | 0.353 | 0.315 |
| red_cards | 24 | 0.294 | 0.734 | -0.177 | 0.168 |
