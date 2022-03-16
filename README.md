# AutoImputer
A way to impute missing values using a Streamlit app

# TODO:

- [x] Add input file
    - [x] Tests for input file shapes and dtypes (only allow numerical for now)
- [x] Add output save option
- [x] Implement validation data error
- [x] clean up simpleImputer (no target cols, only all)
    - [x] confirm that it works (cat cols will get wrong answers _always_, thats why its "simple")
- [x] clean up measure error (no target cols)
- [x] clean up app.py, no target cols
- [x] add selection for user to select the numeric/categorical columns
    - Can't continue without selection
- [x] add data preprocessing function that labels categorical columns
- [x] Bring xgbimputer to app.py
- [ ] Comment all functions and classes, with input parameter specs, clean up useless comments
- [ ] Unit tests for xgboost, random samples with mixed dtypes, inf, negatives, etc...
- [ ] Visualize results/val_errors using plotly
- [ ] Github README
- [ ] pytest and profiling
    - [Can we make XGBoost faster?]
- [ ] Make everything nice looking
- [ ] Add to portfolio


