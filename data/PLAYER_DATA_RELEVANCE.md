# Player-Data Dataset Relevance

This documents the external player/results datasets considered for adding
player-strength signals to the World Cup match-prediction model, what each
contains, whether it was used, and why.

## Datasets considered

| Dataset | Source | Contents | USED | Reason |
|---|---|---|---|---|
| `stefanoleone992/fifa-23-complete-player-dataset` (`male_players.csv`, 5.6 GB) | Kaggle (kagglehub) | Per-player FIFA ratings (overall, potential, value, age, positions) across editions FIFA 15-23, point-in-time via `fifa_version` + `fifa_update_date`. `nation_position` flags the FIFA-designated national-team squad/XI. | **Yes** | Only dataset here that gives point-in-time international-squad player strength. Extracted to `player_ratings_international.csv` (national-team players only, earliest update per edition). Full 5.6 GB file kept in kagglehub cache, not copied into repo. |
| `data/results.csv` (already in repo) | Repo (existing) | 49,477 international match results 1872-2024 with a `tournament` column. | **Yes (already present)** | Confirmed it contains `UEFA Euro` (388 matches, 1960-2024 incl. 2016/2021/2024) and `Copa América` (869 matches, 1916-2024 incl. 2015/2016/2019/2021/2024). No additional results dataset needed for a Euro + Copa América test set. |
| `hugomathien/soccer` (European Soccer Database) | Kaggle | Club match lineups, betting odds, and FIFA player/team attributes 2008-2016. | No | Considered but dropped. Club-centric (domestic leagues), and our task is *international* matches. The national XI we need is already available via `nation_position` in the FIFA dataset, and club lineups / club betting odds don't transfer to national-team matches. Not worth the download. |
| StatsBomb / Wyscout event data | Various | Detailed per-event match data (passes, shots, xG), mostly club competitions. | No | Considered but dropped. Out of scope: club-centric, heavyweight, and far more granular than the match-level prediction model needs. Not trivially useful for international match outcome prediction. |

## FIFA dataset coverage (inventory)

Total rows scanned: 10,003,590. Versions 15-23 present.

| fifa_version | fifa_update_date range | # nationalities | players w/ non-null nation_position | total rows |
|---|---|---|---|---|
| 15 | 2014-08-29 → 2015-09-10 | 157 | 63,713 | 962,549 |
| 16 | 2015-08-28 → 2016-09-22 | 168 | 64,010 | 986,175 |
| 17 | 2016-08-25 → 2017-08-24 | 167 | 98,362 | 1,601,431 |
| 18 | 2017-08-28 → 2018-09-12 | 173 | 96,599 | 1,512,360 |
| 19 | 2018-07-19 → 2019-06-24 | 169 | 75,053 | 1,229,986 |
| 20 | 2019-08-20 → 2020-09-23 | 169 | 68,432 | 1,128,336 |
| 21 | 2020-08-24 → 2021-09-25 | 168 | 71,775 | 1,197,628 |
| 22 | 2021-08-16 → 2022-07-18 | 168 | 48,401 | 1,218,451 |
| 23 | 2022-09-01 → 2023-01-13 | 165 | 7,245 | 166,674 |

(`nation_position` counts are across ALL updates of a version, so they include
the same player at multiple update points. The extract below de-duplicates to a
single pre-season snapshot per edition.)

## Extracted file: `data/player_ratings_international.csv`

- Shape: **9,337 rows × 11 columns**, ~0.78 MB.
- One pre-season national-squad snapshot per FIFA edition: for each
  `fifa_version` the **earliest `fifa_update`** is taken (all editions: update 1),
  keeping only rows with a non-null `nation_position` (i.e. FIFA national-team
  squad members).
- Columns: `fifa_version, fifa_update_date, nationality_name, short_name,
  long_name, player_positions, nation_position, overall, potential, value_eur, age`.
- 56 distinct national teams represented.
- Rows per edition: FIFA15 1081, 16 1104, 17 1081, 18 1149, 19 1104, 20 1127,
  21 1127, 22 759, 23 805 (snapshot dates = the earliest update date above per
  version, e.g. FIFA15 = 2014-08-29, FIFA23 = 2022-09-01).

## Euro / Copa América editions covered by player data

FIFA player snapshots span editions 15-23 (pre-season ~Aug 2014 through Sep 2022).
A tournament is "covered" if a FIFA edition's pre-season squad snapshot exists in
the window leading up to it.

| Tournament edition | Player data available? | Notes |
|---|---|---|
| UEFA Euro 2016 | **Yes** | FIFA16 (snapshot 2015-08-28) / FIFA17 (2016-08-25) bracket it. |
| UEFA Euro 2020 (played 2021) | **Yes** | FIFA21 snapshot 2020-08-24 is a clean pre-tournament snapshot. |
| UEFA Euro 2024 | **No** | Outside FIFA coverage; latest edition (FIFA23) pre-season is 2022-09-01. Gap. |
| Copa América 2015 | **Yes** | FIFA15 snapshot 2014-08-29. |
| Copa América 2016 (Centenario) | **Yes** | FIFA16 (2015-08-28). |
| Copa América 2019 | **Yes** | FIFA19 (2018-07-19) / FIFA20 (2019-08-20). |
| Copa América 2021 | **Yes** | FIFA21 (2020-08-24) / FIFA22 (2021-08-16). |
| Copa América 2024 | **No** | Outside FIFA coverage. Gap (same as Euro 2024). |

Net: a Euro + Copa América test set restricted to **2014/15 through ~2022** is
well-covered by player ratings; **2024 editions of both tournaments are not
covered** by the FIFA dataset.

## Team-name mismatches to flag (for later normalization)

The FIFA `nationality_name` uses some labels that differ from `results.csv` team
names. Normalization is handled later by the model, but the notable cases are:

- `China PR` (FIFA) vs `China` (results.csv)
- `Côte d'Ivoire` (FIFA) vs `Ivory Coast` (results.csv)
- `Korea Republic` (FIFA) vs `South Korea` (results.csv)
- Also present in FIFA in odd-but-matching forms: `Czech Republic`,
  `Republic of Ireland`. Watch for `IR Iran` / `Korea DPR` style labels if more
  nations are pulled in later (not present in the current 56-nation extract).
