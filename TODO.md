# TODO:
## This is a list of tasks that should be done at some point, and some free form comments regarding them    
### The completed tasks are left for audience to see what has been a point of interest and the problems regarding some tasks.    

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
    -  Feature shape mismatch, expected: 6, got 7  -> dummifying was not working as expected -> pd.dummies -> onehotencoder.
    -  contains infs (in categorical col?) --> This is low prio -> Fixed, didnt use replace_infs before...
    -  categorical dummifying not working -> need fix asap --> not working
- [x] simpleimputer categorical support
- [x] categorical data measure error
- [x] cache xgboost results
- [x] add pure numeric and pure categorical test_dataset to test_suite
- [x] XGBImputer has some WETWET code when wrangling the x_train,x_test data (dummifying), fix this
- [x] 'format_dtypes" function: reformat to only give dtypes_list. Do not change data
- [x] Test if input data has rows with all missing values -> remove from data + warning
- [x] profile code (own .py file for imputing xgb, try to speedup) -> did some speedup, but gridsearch is a huge bottleneck
- [x] Visualize data preview
- [x] Add warning if some column completely empty -> imputing wont work.
    - --> drop column
- [x] Make everything nice looking
    - [x] Add intoduction
    - [x] Add explanations using st.expander to everything
- [x] get test data from kaggle/oecd/eu databank etc..
- [x] clean up code (commented unused code etc..)
- [x] add warning if all rows contain atleast one missing value -> cant validate
- [x] add dtype inference
- [x] Add logo/figure of "AutoImputer" to readme and app.py
- [x] Github README
- [ ] ~~Datawig model?~~
    - Needs C++ 14+ and seems to not be for me, even though c++ build tools already installed.
    - Better not to waste time with this
- [ ] Validation error computation in app.py needs some complex session_state wrangling.
    - a button press reruns the whole script -> imputing is gone
- [ ] PyTorch LSTM/RNN/CNN/??? Randomforest (https://www.frontiersin.org/articles/10.3389/fdata.2021.693674/full#h6)
    - Read up on what to use
    - Choose one method, implement it
    - tests
    - Label propagation? (Data Mining technique) for inputs that have less than x% labeled data.
    - Cant use same framework (class) with xgboost as other methods do not work withm issing values...
- [ ] Should we train XGBoost with only fully complete data?
    - XGBoost can handle missing values in feature columns
    - --> Will full data give better results?
- [ ] Is XGBoost updated? -> update to get rid of FutureWarning.
    - [ ] Also if available, cateogorical support -> no one-hot encoding!
- [ ] XGBoost is way too slow in the cloud, make it faster somehow.