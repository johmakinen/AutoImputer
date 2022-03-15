# AutoImputer
A way to impute missing values using a Streamlit app

# TODO:

- [x] Add input file
    - [x] Tests for input file shapes and dtypes (only allow numerical for now)
- [x] Add output save option
- [x] Implement validation data error
- [ ] clean up simpleImputer (no target cols, only all)
    - [ ] confirm that it works (cat cols will get wrong answers _always_, thats why its "simple")
- [ ] clean up measure error (no target cols)
- [ ] clean up app.py, no target cols
- [ ] add selection for user to select the numeric/categorical columns
    - Can't continue without selection
- [ ] add data preprocessing function that labels categorical columns
- [ ] Bring xgbimputer to app.py
- [ ] Comment all functions and classes, with input parameter specs
- [ ] Unit tests for xgboost, random samples with mixed dtypes, inf, negatives, etc...
- [ ] Visualize results/val_errors using plotly
- [ ] Github README
- [ ] pytest and profiling
- [ ] Make everything nice looking
- [ ] Add to portfolio


