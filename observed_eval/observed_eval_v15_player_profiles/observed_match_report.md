# v15_catboost_model observed World Cup evaluation

- Matches: 12
- Result accuracy: 58.3%
- Result log loss: 1.045
- Three-way Brier score: 0.637
- Ranked probability score: 0.201
- Goal MAE: 1.010
- Goal RMSE: 1.499
- Exact-score accuracy: 25.0%
- Log-loss skill vs uniform forecast: 4.9%

Bootstrap intervals measure sampling uncertainty across this observed match set. With a small number of matches, they should be expected to be wide.

## Subgroups

| dimension | subgroup | n_matches | result_accuracy | mean_result_log_loss | mean_result_brier | mean_result_rps | mean_goal_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 12 | 0.583 | 1.045 | 0.637 | 0.201 | 1.010 |
| actual_result | draw | 4 | 0.250 | 1.504 | 0.966 | 0.180 | 0.556 |
| actual_result | team_a_win | 7 | 0.714 | 0.858 | 0.506 | 0.227 | 1.297 |
| actual_result | team_b_win | 1 | 1.000 | 0.520 | 0.247 | 0.099 | 0.813 |
| stage | Group Stage | 12 | 0.583 | 1.045 | 0.637 | 0.201 | 1.010 |
| group | Group A | 2 | 1.000 | 0.803 | 0.462 | 0.206 | 0.624 |
| group | Group B | 2 | 0.000 | 1.516 | 0.996 | 0.193 | 0.649 |
| group | Group C | 2 | 0.500 | 1.007 | 0.602 | 0.139 | 0.652 |
| group | Group D | 2 | 0.500 | 1.091 | 0.682 | 0.316 | 1.367 |
| group | Group E | 2 | 0.500 | 0.736 | 0.418 | 0.185 | 1.599 |
| group | Group F | 2 | 1.000 | 1.118 | 0.664 | 0.168 | 1.165 |

## Box events

| event | n_team_observations | mae | rmse | bias | correlation |
| --- | --- | --- | --- | --- | --- |
| shots | 24 | 5.161 | 6.309 | 1.028 | 0.393 |
| shots_on_target | 24 | 1.948 | 2.283 | 0.694 | 0.563 |
| possession | 24 | 7.026 | 8.323 | 0.000 | 0.771 |
| fouls | 24 | 5.054 | 5.860 | 3.175 | 0.107 |
| yellow_cards | 24 | 1.139 | 1.295 | 0.353 | 0.315 |
| red_cards | 24 | 0.294 | 0.734 | -0.177 | 0.168 |
