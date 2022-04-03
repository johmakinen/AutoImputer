<div align="center">

![Streamlit AutoImputer](figures/Logo_figma.png)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/johmakinen/autoimputer/main/app.py)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

</div>

![AutoImputer video](figures\app_video.gif)




## 📈 Usage

The app has (at the moment) two options for imputing the values. In the simplest terms, you just give your data as a csv, select the preferred options and let the app fill the missing values.

* <strong>Input your data</strong>: The app read the data and suggests the columntypes for you.
* <strong>Confirm the selected column types</strong>: This is important as classifying a numeric feature wont work.
* <strong>Select model</strong>: Simple models are much faster but can be less accurate. Complex models can be too slow and result in a timeout. All depends on the size and complexity of your data. With over 10k rows the XGBoost imputer will probably not work (and I have no patience to wait for hours.)
* <strong>Impute</strong>: Impute the values with the chosen model and settings.
* <strong>Check validation metrics</strong>: These metrics give a rough estimate how well the imputed values match the data.
Once you are satisfied, you can to download the imputed dataset.

![Autoimputer tbl](figures\table_fig.png)

## 🛠️ How to contribute ?

All contributions, ideas and bug reports are welcome! 
It is encouraged to open an [issue](https://github.com/johmakinen/AutoImputer/issues) for any change you would like to make on this project.
There is also a [TODO](https://github.com/johmakinen/AutoImputer/blob/main/TODO.md) list where you can find interesting future improvements and problems that are just waiting to be implemented!





