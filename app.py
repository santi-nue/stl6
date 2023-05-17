# LIBRARIES
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import zscore,shapiro,probplot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors,acorr_breusch_godfrey
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pickle import dumps
import streamlit as st

# CONTENT
st.markdown("<h1 style='text-align: center;'>Linear Regression</h1>",unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Made by <b><a href='https://www.linkedin.com/in/mathewdarren/'>Mathew Darren Kusuma</a></b></p>",
    unsafe_allow_html=True
)
st.image(Image.open("robot-handshake-human-background-futuristic-digital-age.jpg"))
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

st.sidebar.markdown('''
# Navigation
- [Data](#data)
- [Assumption Tests](#assumption-tests)
- [Significance Tests](#significance-tests)
- [Linear Regression Model](#linear-regression-model)
- [Download](#download)
''',unsafe_allow_html=True)

# DATA
st.header("Data")
file = st.file_uploader("Upload your file in Excel format",type=["csv","xlsx"])
if file is not None:
    try:
        df = pd.read_csv(file)
    except:
        df = pd.read_excel(file)
    st.subheader("Data Preview")
    rows = st.slider("Show the first n rows",min_value=1,max_value=len(df),value=5,key="rows")
    st.dataframe(df.head(rows))

    z_data = st.checkbox("Data standardization")
    if z_data:
        df = df.apply(zscore,ddof=1)
        df.columns = [f"ZScore({i})" for i in df.columns]
        st.dataframe(df.head(rows))

if file is None:
    sample_data = st.checkbox("Use a sample data")
    if sample_data:
        df = pd.read_excel("sample_data.xlsx")
        st.subheader("Data Preview")
        rows = st.slider("Show the first n rows",min_value=1,max_value=len(df),value=5,key="rows")
        st.dataframe(df.head(rows))

        z_data = st.checkbox("Data standardization")
        if z_data:
            df = df.apply(zscore,ddof=1)
            df.columns = [f"ZScore({i})" for i in df.columns]
            st.dataframe(df.head(rows))

if (file is not None) or (sample_data):
    # Choose independent and dependent variables
    st.subheader("Independent and Dependent Variables")

    # Dependent
    dependent = st.selectbox("Select one dependent variable",df.columns)
    y = df[dependent]

    # Independent
    independent = st.multiselect(
        "Select one or more independent variables",
        ["All"] + [i for i in df.columns if i != dependent]
    )
    if "All" in independent:
        independent = [i for i in df.columns if i != dependent]
    X = df[independent]

    if (dependent) and (independent):
        st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

        # ASSUMPTIONS
        st.header("Assumption Tests")

        # Model
        model = sm.OLS(y,sm.add_constant(X)).fit()
        model2 = LinearRegression().fit(X,y)

        # Linearity
        st.subheader("Linearity")
        resid = list(model.resid)
        zresid = zscore(resid,ddof=1)
        pred = list(model.predict(sm.add_constant(X)))
        zpred = zscore(pred,ddof=1)

        with sns.axes_style("darkgrid"):
            fig,ax = plt.subplots(figsize=(10,5),dpi=200)
            ax.scatter(zpred,zresid,color="blue")
            ax.set_title(f"Scatter Plot\nDependent Variable: {dependent}")
            ax.set_xlabel("Regression Standardized Predicted Value")
            ax.set_ylabel("Regression Standardized Residual")
        st.pyplot(fig)

        # Normality
        st.subheader("Normality")
        st.markdown("<h4>Visual</h4>",unsafe_allow_html=True)

        with sns.axes_style("darkgrid"):
            fig,ax = plt.subplots(figsize=(10,5),dpi=200,nrows=1,ncols=2)

            probplot(resid,plot=ax[0])
            ax[0].set_title("Normal Q-Q Plot")
            ax[0].set_xlabel("Expected Normal")
            ax[0].set_ylabel("Observed Value")

            sns.histplot(resid,color="blue",kde=True,ax=ax[1])
            ax[1].lines[0].set_color("red")
            ax[1].set_title("Histogram")
            ax[1].set_xlabel("Unstandardized Residual")
            ax[1].set_ylabel("Frequency")

            plt.suptitle("Unstandardized Residual")
        st.pyplot(fig)

        st.markdown("<h4>Formal</h4>",unsafe_allow_html=True)

        # Function to gives colors
        def highlight_cells(val):
            if val == "TRUE":
                color = "#6AAB9C"
            elif val == "FALSE":
                color = "#E06C78"
            else:
                color = ""
            return "background-color: {}".format(color)

        ks_stat,ks_pval = lilliefors(resid,dist="norm",pvalmethod="approx")
        sw_stat,sw_pval = shapiro(resid)

        normality = pd.DataFrame(
            {
                "Statistic":[ks_stat,sw_stat],
                "P-Value":[ks_pval,sw_pval]
            },
            index=["Kolmogorov-Smirnov","Shapiro-Wilk"]
        )
        normality["Normal"] = ["TRUE" if i >= 0.05 else "FALSE" for i in normality["P-Value"]]
        st.dataframe(normality.style.applymap(highlight_cells,subset=["Normal"]))

        # Homoscedasticity
        st.subheader("Homoscedasticity")
        st.markdown("<h4>Visual</h4>",unsafe_allow_html=True)

        sresid = list(model.outlier_test()["student_resid"])

        with sns.axes_style("darkgrid"):
            fig,ax = plt.subplots(figsize=(10,5),dpi=200)
            ax.scatter(zpred,sresid,color="blue")
            ax.set_title(f"Scatter Plot\nDependent Variable: {dependent}")
            ax.set_xlabel("Regression Standardized Predicted Value")
            ax.set_ylabel("Regression Studentized Residual")
        st.pyplot(fig)

        st.markdown("<h4>Formal</h4>",unsafe_allow_html=True)
        abs_resid_model = sm.OLS(abs(np.array(resid)),sm.add_constant(X)).fit()
        homoscedasticity = abs_resid_model.summary2().tables[1][["t","P>|t|"]]
        homoscedasticity.columns = ["t","P-Value"]
        homoscedasticity = homoscedasticity.rename(index={"const":"Constant"})
        homoscedasticity["Homoscedasticity"] = [""] + ["TRUE" if i >= 0.05 else "FALSE" for i in homoscedasticity["P-Value"].iloc[1:]]
        st.dataframe(homoscedasticity.style.applymap(highlight_cells,subset=["Homoscedasticity"]))

        # Non-Autocorrelation
        st.subheader("Non-Autocorrelation")
        non_autocorrelation = pd.DataFrame(
            {
                "Statistic":acorr_breusch_godfrey(model,nlags=2)[0],
                "P-Value":acorr_breusch_godfrey(model,nlags=2)[1],
            },
            index=["Breusch-Godfrey"]
        )
        non_autocorrelation["Non-Autocorrelation"] = ["TRUE" if i >= 0.05 else "FALSE" for i in non_autocorrelation["P-Value"]]
        st.dataframe(non_autocorrelation.style.applymap(highlight_cells,subset=["Non-Autocorrelation"]))

        # Non-Multicollinearity
        if len(independent) > 1:
            st.subheader("Non-Multicollinearity")
            X["intercept"] = 1
            vif = pd.DataFrame(index=X.columns)
            vif["VIF"] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
            vif.drop("intercept",inplace=True)
            vif["Non-Multicollinearity"] = ["TRUE" if i < 10 else "FALSE" for i in vif["VIF"]]
            X.drop("intercept",axis=1,inplace=True)
            st.dataframe(vif.style.applymap(highlight_cells,subset=["Non-Multicollinearity"]))

        st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

        # SIGNIFICANCE
        st.header("Significance Tests")

        # F-Test
        st.subheader("F-Test")
        f_test = pd.DataFrame(
            {
                "F":[model.fvalue],
                "P-Value":[model.f_pvalue]
            },
        index=["Regression"]
        )
        f_test["Significance"] = ["TRUE" if i < 0.05 else "FALSE" for i in f_test["P-Value"]]
        st.dataframe(f_test.style.applymap(highlight_cells,subset=["Significance"]))

        # t-Test
        st.subheader("t-Test")
        t_test = model.summary2().tables[1][["t","P>|t|"]]
        t_test.columns = ["t","P-Value"]
        t_test = t_test.rename(index={"const":"Constant"})
        t_test["Significance"] = [""] + ["TRUE" if i < 0.05 else "FALSE" for i in t_test["P-Value"].iloc[1:]]
        st.dataframe(t_test.style.applymap(highlight_cells,subset=["Significance"]))

        # Coefficient of Determination
        st.subheader("Coefficient of Determination")
        r = pd.DataFrame(
            data=[np.sqrt(model.rsquared),model.rsquared,model.rsquared_adj],
            index=["R","R-Squared","Adjusted R-Squared"],
            columns=["Value"]
        )
        st.dataframe(r)

        st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

        # LINEAR REGRESSION MODEL
        st.header("Linear Regression Model")

        # Equation
        st.subheader("Model Equation")
        coef = pd.DataFrame(model.params,columns=["Unstandardized Coefficient"])
        coef = coef.rename(index={"const":"Constant"})
        st.dataframe(coef)

        round_up = st.slider("Round with n digits after decimal point",min_value=1,max_value=10,value=6,key="round_up")
        x_symbol = [sp.Symbol(f"({i})") for i in model.params.index[1:]]
        y_symbol = sp.Symbol(f"\hat{{{dependent}}}")
        sums = np.round(model.params.values[0],round_up)
        for i in range(len(X.columns)):
            sums += x_symbol[i] * np.round(model.params.values[1:][i],round_up)
        st.write(fr"$ {y_symbol} = {sums} $".replace("*",""))

        # Results
        st.subheader("Results")
        rows2 = st.slider("Show the first n rows",min_value=1,max_value=len(df),value=5,key="rows2")
        
        col1,col2 = st.columns(2)
        with col1:
            results = pd.DataFrame({
                "Actual":y,
                "Prediction":pred,
                "| Residual | ":abs(np.array(resid))
            })
            st.dataframe(results.head(rows2))
        
        with col2:
            metrics = pd.DataFrame(
                data=[mean_absolute_error(y,pred),np.sqrt(mean_squared_error(y,pred))],
                index=["Mean Abosulute Error","Root Mean Squared Error"],
                columns=["Metric"]
            )
            st.dataframe(metrics)

        st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

        # DOWNLOAD
        st.header("Download")
        model_download = st.download_button(
            "Download the linear regression model",
            data=dumps(model2),
            file_name="linear_regression_model.pkl"
        )
        if model_download:
            st.success("The linear regression model has been downloaded!")