# v15_catboost_model observed World Cup evaluation

- Matches: 16
- Result accuracy: 43.8% (95% CI 18.8% to 68.8%)
- Result log loss: 1.145 (95% CI 0.908 to 1.372)
- Three-way Brier score: 0.729 (95% CI 0.546 to 0.902)
- Ranked probability score: 0.201 (95% CI 0.152 to 0.257)
- Goal MAE: 0.950 (95% CI 0.639 to 1.315)
- Goal RMSE: 1.488 (95% CI 0.916 to 1.976)
- Exact-score accuracy: 31.2%
- Log-loss skill vs uniform forecast: -4.3%

Bootstrap intervals measure sampling uncertainty across this observed match set. With a small number of matches, they should be expected to be wide.

## Subgroups

| dimension | subgroup | n_matches | result_accuracy | mean_result_log_loss | mean_result_brier | mean_result_rps | mean_goal_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 16 | 0.438 | 1.145 | 0.729 | 0.201 | 0.950 |
| actual_result | draw | 8 | 0.000 | 1.481 | 0.997 | 0.204 | 0.830 |
| actual_result | team_a_win | 7 | 0.857 | 0.843 | 0.484 | 0.211 | 1.118 |
| actual_result | team_b_win | 1 | 1.000 | 0.581 | 0.292 | 0.117 | 0.742 |
| stage | Group Stage | 16 | 0.438 | 1.145 | 0.729 | 0.201 | 0.950 |
| group | Group A | 2 | 1.000 | 0.519 | 0.253 | 0.096 | 0.227 |
| group | Group B | 2 | 0.000 | 1.442 | 0.974 | 0.196 | 0.604 |
| group | Group C | 2 | 0.500 | 0.912 | 0.537 | 0.127 | 0.512 |
| group | Group D | 2 | 1.000 | 0.965 | 0.576 | 0.249 | 1.177 |
| group | Group E | 2 | 0.500 | 0.975 | 0.570 | 0.262 | 1.552 |
| group | Group F | 2 | 0.500 | 1.170 | 0.718 | 0.203 | 1.319 |
| group | Group G | 2 | 0.000 | 1.443 | 0.975 | 0.196 | 0.526 |
| group | Group H | 2 | 0.000 | 1.738 | 1.226 | 0.281 | 1.684 |

## Box events

| event | n_team_observations | mae | rmse | bias | correlation |
| --- | --- | --- | --- | --- | --- |
| shots | 32 | 5.556 | 6.706 | -0.157 | 0.301 |
| shots_on_target | 32 | 2.210 | 2.787 | 0.307 | 0.305 |
| possession | 32 | 6.762 | 8.565 | 0.000 | 0.635 |
| fouls | 32 | 5.325 | 6.327 | 3.458 | -0.002 |
| yellow_cards | 32 | 1.079 | 1.254 | 0.391 | 0.204 |
| red_cards | 32 | 0.228 | 0.636 | -0.125 | 0.194 |
