# v13_live_signal_model observed World Cup evaluation

- Matches: 12
- Result accuracy: 58.3% (95% CI 33.3% to 83.3%)
- Result log loss: 1.041 (95% CI 0.817 to 1.259)
- Three-way Brier score: 0.634 (95% CI 0.471 to 0.793)
- Ranked probability score: 0.200 (95% CI 0.153 to 0.249)
- Goal MAE: 0.996 (95% CI 0.681 to 1.357)
- Goal RMSE: 1.480 (95% CI 0.832 to 2.009)
- Exact-score accuracy: 25.0%
- Log-loss skill vs uniform forecast: 5.3%

Bootstrap intervals measure sampling uncertainty across this observed match set. With a small number of matches, they should be expected to be wide.

## Subgroups

| dimension | subgroup | n_matches | result_accuracy | mean_result_log_loss | mean_result_brier | mean_result_rps | mean_goal_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 12 | 0.583 | 1.041 | 0.634 | 0.200 | 0.996 |
| actual_result | draw | 4 | 0.250 | 1.506 | 0.967 | 0.181 | 0.548 |
| actual_result | team_a_win | 7 | 0.714 | 0.851 | 0.501 | 0.225 | 1.269 |
| actual_result | team_b_win | 1 | 1.000 | 0.509 | 0.239 | 0.096 | 0.876 |
| stage | Group Stage | 12 | 0.583 | 1.041 | 0.634 | 0.200 | 0.996 |
| group | Group A | 2 | 1.000 | 0.788 | 0.451 | 0.200 | 0.563 |
| group | Group B | 2 | 0.000 | 1.516 | 0.996 | 0.194 | 0.675 |
| group | Group C | 2 | 0.500 | 1.004 | 0.601 | 0.138 | 0.634 |
| group | Group D | 2 | 0.500 | 1.082 | 0.676 | 0.314 | 1.377 |
| group | Group E | 2 | 0.500 | 0.739 | 0.420 | 0.186 | 1.565 |
| group | Group F | 2 | 1.000 | 1.116 | 0.662 | 0.166 | 1.165 |

## Box events

| event | n_team_observations | mae | rmse | bias | correlation |
| --- | --- | --- | --- | --- | --- |
| shots | 24 | 5.161 | 6.309 | 1.028 | 0.393 |
| shots_on_target | 24 | 1.948 | 2.283 | 0.694 | 0.563 |
| possession | 24 | 7.026 | 8.323 | 0.000 | 0.771 |
| fouls | 24 | 5.054 | 5.860 | 3.175 | 0.107 |
| yellow_cards | 24 | 1.139 | 1.295 | 0.353 | 0.315 |
| red_cards | 24 | 0.294 | 0.734 | -0.177 | 0.168 |
