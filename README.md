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
- [x] Comment all functions and classes, with input parameter specs, clean up useless comments
- [x] Unit tests for xgboost, random samples with mixed dtypes, inf, negatives, etc...
- [x] XGBoost categorical data when doing regression (categorical feature)
    - [x] imputation when categorical target has nans
    - At the moment, dummifying all but target -> not working looping only over nan cols. Done [x]
- [x] Validation error for each column separately, use st.expander in app.py
- [x] Pytest:
    -  Feature shape mismatch, expected: 6, got 7 
    -  contains infs (in categorical col?) --> This is low prio
    -  categorical dummifying not working -> need fix asap
- [ ] simpleimputer categorical support
- [ ] categorical data measure error
- [ ] cache xgboost results
- [ ] profile code (own .py file for imputing xgb, try to speedup)
- [x] XGBImputer has some WETWET code when wrangling the x_train,x_test data (dummifying), fix this
- [x] 'format_dtypes" function: reformat to only give dtypes_list. Do not change data
- [ ] Test if input data has rows with all missing values -> remove from data + warning
- [ ] Visualize results/val_errors using plotly
- [ ] Github README
- [ ] PyTorch LSTM/RNN/CNN/???
    - Read up on what to use
    - Choose one method, implement it
    - tests
- [ ] Make everything nice looking
    - [ ] Add intoduction
    - [ ] Add explanations using st.expander to everything
    - [ ] Change theme
- [ ] Is XGBoost updated? -> update to get rid of FutureWarning.
    - [ ] Also if available, cateogircal support -> no one-hot encoding!
- [ ] what if contains infs in categorical col --> This is low prio
- [ ] Add to portfolio





