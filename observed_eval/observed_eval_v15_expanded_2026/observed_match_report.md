# v15_catboost_model observed World Cup evaluation

- Matches: 12
- Result accuracy: 58.3% (95% CI 33.3% to 83.3%)
- Result log loss: 0.997 (95% CI 0.754 to 1.235)
- Three-way Brier score: 0.605 (95% CI 0.427 to 0.776)
- Ranked probability score: 0.189 (95% CI 0.131 to 0.256)
- Goal MAE: 0.899 (95% CI 0.577 to 1.258)
- Goal RMSE: 1.401 (95% CI 0.797 to 1.891)
- Exact-score accuracy: 33.3%
- Log-loss skill vs uniform forecast: 9.2%

Bootstrap intervals measure sampling uncertainty across this observed match set. With a small number of matches, they should be expected to be wide.

## Subgroups

| dimension | subgroup | n_matches | result_accuracy | mean_result_log_loss | mean_result_brier | mean_result_rps | mean_goal_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 12 | 0.583 | 0.997 | 0.605 | 0.189 | 0.899 |
| actual_result | draw | 4 | 0.000 | 1.371 | 0.894 | 0.169 | 0.554 |
| actual_result | team_a_win | 7 | 0.857 | 0.843 | 0.484 | 0.211 | 1.118 |
| actual_result | team_b_win | 1 | 1.000 | 0.581 | 0.292 | 0.117 | 0.742 |
| stage | Group Stage | 12 | 0.583 | 0.997 | 0.605 | 0.189 | 0.899 |
| group | Group A | 2 | 1.000 | 0.519 | 0.253 | 0.096 | 0.227 |
| group | Group B | 2 | 0.000 | 1.442 | 0.974 | 0.196 | 0.604 |
| group | Group C | 2 | 0.500 | 0.912 | 0.537 | 0.127 | 0.512 |
| group | Group D | 2 | 1.000 | 0.965 | 0.576 | 0.249 | 1.177 |
| group | Group E | 2 | 0.500 | 0.975 | 0.570 | 0.262 | 1.552 |
| group | Group F | 2 | 0.500 | 1.170 | 0.718 | 0.203 | 1.319 |

## Box events

| event | n_team_observations | mae | rmse | bias | correlation |
| --- | --- | --- | --- | --- | --- |
| shots | 24 | 5.161 | 6.309 | 1.028 | 0.393 |
| shots_on_target | 24 | 1.948 | 2.283 | 0.694 | 0.563 |
| possession | 24 | 7.026 | 8.323 | 0.000 | 0.771 |
| fouls | 24 | 5.054 | 5.860 | 3.175 | 0.107 |
| yellow_cards | 24 | 1.139 | 1.295 | 0.353 | 0.315 |
| red_cards | 24 | 0.294 | 0.734 | -0.177 | 0.168 |
