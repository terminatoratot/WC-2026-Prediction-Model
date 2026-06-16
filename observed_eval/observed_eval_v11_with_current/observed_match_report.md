# V11 observed World Cup evaluation

- Matches: 8
- Result accuracy: 50.0% (95% CI 12.5% to 87.5%)
- Result log loss: 1.104 (95% CI 0.839 to 1.366)
- Three-way Brier score: 0.685 (95% CI 0.484 to 0.885)
- Ranked probability score: 0.212 (95% CI 0.159 to 0.273)
- Goal MAE: 0.812 (95% CI 0.569 to 1.126)
- Goal RMSE: 1.019 (95% CI 0.668 to 1.384)
- Exact-score accuracy: 37.5%
- Log-loss skill vs uniform forecast: -0.5%

Bootstrap intervals measure sampling uncertainty across this observed match set. With a small number of matches, they should be expected to be wide.

## Subgroups

| dimension | subgroup | n_matches | result_accuracy | mean_result_log_loss | mean_result_brier | mean_result_rps | mean_goal_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 8 | 0.500 | 1.104 | 0.685 | 0.212 | 0.812 |
| actual_result | draw | 3 | 0.000 | 1.528 | 0.995 | 0.191 | 0.580 |
| actual_result | team_a_win | 4 | 0.750 | 0.935 | 0.563 | 0.257 | 0.970 |
| actual_result | team_b_win | 1 | 1.000 | 0.509 | 0.239 | 0.096 | 0.876 |
| stage | Group Stage | 8 | 0.500 | 1.104 | 0.685 | 0.212 | 0.812 |
| group | Group A | 2 | 1.000 | 0.788 | 0.451 | 0.200 | 0.563 |
| group | Group B | 2 | 0.000 | 1.542 | 1.012 | 0.197 | 0.675 |
| group | Group C | 2 | 0.500 | 1.004 | 0.601 | 0.138 | 0.634 |
| group | Group D | 2 | 0.500 | 1.082 | 0.676 | 0.314 | 1.377 |

## Box events

| event | n_team_observations | mae | rmse | bias | correlation |
| --- | --- | --- | --- | --- | --- |
| shots | 16 | 6.082 | 7.142 | 0.755 | 0.163 |
| shots_on_target | 16 | 1.843 | 2.204 | 0.675 | 0.382 |
| possession | 16 | 8.376 | 9.556 | 0.000 | 0.739 |
| fouls | 16 | 4.250 | 5.156 | 2.250 | 0.358 |
| yellow_cards | 16 | 1.076 | 1.293 | 0.203 | 0.267 |
| red_cards | 16 | 0.408 | 0.895 | -0.298 | 0.199 |
