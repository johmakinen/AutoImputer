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
- [ ] categorical data measure error
- [ ] cache xgboost results
- [ ] XGBImputer has some WETWET code when wrangling the x_train,x_test data (dummifying), fix this
- [ ] 'format_dtypes" function: reformat to only give dtypes_list. Do not change data
- [ ] Test if input data has rows with all missing values -> remove from data + warning
- [ ] Visualize results/val_errors using plotly
- [ ] Github README
- [ ] Can we make XGBoost faster?
- [ ] Make everything nice looking
    - [ ] Add intoduction
    - [ ] Add explanations using st.expander to everything
    - [ ] Change theme
- [ ] Add to portfolio


