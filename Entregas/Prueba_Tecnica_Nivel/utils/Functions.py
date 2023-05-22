# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder

# Function to obtain an univariant analysis


def get_univariate_analysis(df, df_no_outliers=None):
    """
    Obtiene el análisis univariante de un dataframe

    args:
        list_series: lista de series o dataframes

    returns:
        DataFrame con el análisis univariante de cada columna
    """
    if df_no_outliers is None:
        df_no_outliers = df

    normal_var = 0
    no_normal_var = 0
    univar_analysis = pd.DataFrame(
        {}, columns=["Media", "Mediana", "Moda", "Varianza", "Desviacion_estandar", "Percentil_25", "Percentil_75", "K_test", "p_value", "Distribución"])

    for col in df.columns:
        print(f"\033[1mAnálisis univariante de {col}:\033[0m")
        # Realiza un análisis si la variable es categórica
        if (df[col].dtype == object) or ((col == "Codigo_municipio") or (col == "Codigo_provincia")):
            print(f"Variable categórica:")
            print(f"-Valores únicos:\n{df[col].value_counts()}")
            print(f"-Número de valores únicos: {df[col].nunique()}")
            print("\n\n\n")

        # Realiza un análisis si la variable es numérica
        else:
            # Crea un histograma y un boxplot de la variable
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(df_no_outliers[col], kde=True, ax=axes[0])
            axes[0].set_title("Histograma")
            sns.boxplot(df_no_outliers[col], ax=axes[1])
            axes[1].set_title("Boxplot")
            fig.suptitle(f"Análisis de {col}")
            plt.show()

            # Comprueba estadisticamente con el test Kolmogorov-Smirnov si la variable sigue una distribución normal
            stat, p = ss.kstest(df[col], 'norm')
            alpha = 0.05

            # Añade los datos al DataFrame dependiendo de si se acepta H0 o no
            if p < alpha:
                no_normal_var += 1
                univar_analysis = univar_analysis.append({"Municipio": col, "Media": df[col].mean(), "Mediana": df[col].median(
                ), "Moda": df[col].mode().iloc[0], "Varianza": df[col].var(), "Desviacion_estandar": df[col].std(), "Percentil_25": df[col].quantile(0.25), "Percentil_75": df[col].quantile(0.75), "K_test": stat, "p_value": p, "Distribución": "No normal"}, ignore_index=True) # type: ignore
                print(
                    f"La columna {col} no presenta una distribución normal\n\n\n")

            else:
                normal_var += 1
                univar_analysis = univar_analysis.append({"Municipio": col, "Media": df[col].mean(), "Mediana": df[col].median(
                ), "Moda": df[col].mode().iloc[0], "Varianza": df[col].var(), "Desviacion_estandar": df[col].std(), "Percentil_25": df[col].quantile(0.25), "Percentil_75": df[col].quantile(0.75), "K_test": stat, "p_value": p, "Distribución": "Normal"}, ignore_index=True) # type: ignore
                print(
                    f"La columna {col} presenta una distribución normal\n\n\n")

    # Establece la columna Municipio como índice
    univar_analysis.set_index("Municipio", inplace=True)

    # Imprime el número de variables que siguen una distribución normal y el que no
    print(
        f"\033[1mNúmero de variables que siguen una distribución normal:\033[0m {normal_var}")
    print(
        f"\033[1mNúmero de variables que no siguen una distribución normal:\033[0m {no_normal_var}")
    return univar_analysis


# Function to obtain bivariate analysis
def get_bivariate_analysis(df, annot=False):
    """
    Obtiene el análisis bivariante de un dataframe

    args:
        list_series: lista de series o dataframes
        annot: boolean, si se quiere mostrar la correlación entre las variables
    """
    if annot:
        plt.figure(figsize=(25, 25))
        sns.heatmap(df.corr(), annot=True)
    else:
        sns.heatmap(df.corr(), annot=False)
    sns.pairplot(df, diag_kind='kde')


# Detecta outliers de un dataframe
def get_outliers(df):
    """
    Detecta outliers de un conjunto de series o de un dataframe

    args:
        list_series: lista de series o dataframes

    reutrns:
        lista de diccionarios. Primer diccionario con la cantidad de outliers por columna.
                                Segundo diccionario con la lista de outliers por columna.

    """
    outliers = {}
    outliers_len = {}
    for col in df.columns:
        if (df[col].dtype == "object") or (df[col].dtype == "category") or (df[col].dtype == "bool"):
            pass
        else:
            outliers_col = []
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            minimum = Q1 - (1.5 * IQR)
            maximum = Q3 + (1.5 * IQR)
            for data in df[col]:
                if data < minimum or data > maximum:
                    outliers_col.append(data)
            outliers[col] = outliers_col
            outliers_len[col] = len(outliers_col)
    output = [outliers_len, outliers]
    return output

