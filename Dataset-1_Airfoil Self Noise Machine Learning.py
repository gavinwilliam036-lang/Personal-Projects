import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# ----- DATA PREPARATION -----
df_0 = pd.read_csv('C:/Users/gavin/OneDrive/Documents/ITB/Semester - 4/WI2002 LiDIA/AirfoilSelfNoise.csv')

df_1 = df_0.drop_duplicates()

Q1 = df_1.quantile(0.25)
Q3 = df_1.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = df_1[(df_1>=lower_bound) & (df_1<=upper_bound)].dropna()

# ----- PROGRAM FRONTEND -----
print('SELECT PROGRAM MODE')
print('1. Data Visualization')
print('2. Feature Correlation Map')
print('3. Machine Learning')
mode = input('Choose program mode (1-3): ')

if mode == '1': 
    # DATA VISUALIZATION
    # Check exactly how many missing values are in each column
    print('\nmissing values in the original dataset: ')
    print(df_0.isnull().sum())
    print('\nmissing values in the cleaned dataset: ')
    print(df_clean.isnull().sum())
    sns.set_theme(style="whitegrid")

    print('\nSELECT DATA SETS VISUALIZATION: ')
    print('1. Original')
    print('2. Cleaned')
    print('3. Both')
    data_type = input('Choose data type (1/2/3): ')

    data_type_desc = ['Original', 'Cleaned', 'Both']
    desc = data_type_desc[int(data_type) - 1]

    def plot_script(df_0, df_clean, variables, axes, mode='3'):
        for i, var in enumerate(variables):
            master_bins = np.histogram_bin_edges(df_0[var], bins=30)

            if mode == '1' or mode == '3':
                sns.histplot(data=df_0, x=var, color='red', kde=True, label='Original', ax=axes[i], bins=master_bins)
            if mode == '2' or mode == '3':
                sns.histplot(data=df_clean, x=var, color='blue', kde=True, label='Cleaned', ax=axes[i], bins=master_bins)
        
            axes[i].set_title(f'Distribution of {var} from {desc} Dataset')
            axes[i].set_xlabel(f'{var}')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            
    variables = df_0.select_dtypes(include=['float64', 'int64']).columns
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    axes = axes.flatten()

    plot_script(df_0, df_clean, variables, axes, mode=data_type)

    plt.tight_layout()
    plt.show()
    
    # Create multiple figures, each with 1 row and 2 columns
    for i in range(0, len(variables), 2):
        chunk = variables[i:i+2]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        axes = axes.flatten()
        plot_script(df_0, df_clean, chunk, axes, mode=data_type)
        plt.tight_layout()
        plt.show()
elif mode == '2':
    # ----- FEATURE CORRELATION CHECK -----
    X = df_clean.drop('SSPL', axis=1)
    y = df_clean['SSPL']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    train_data = X_train.copy()
    train_data['SSPL'] = y_train

    correlation_matrix_pearson = train_data.corr(method='pearson')
    correlation_matrix_spearman = train_data.corr(method='spearman')

    sspl_corr_pearson = correlation_matrix_pearson['SSPL'].drop('SSPL').sort_values(ascending=False)
    sspl_corr_spearman = correlation_matrix_spearman['SSPL'].drop('SSPL').sort_values(ascending=False)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # It's good practice to print or display the sorted correlations themselves
    print("Pearson Correlation with SSPL (sorted):")
    print(sspl_corr_pearson)
    print("\nSpearman Correlation with SSPL (sorted):")
    print(sspl_corr_spearman)

    # Plotting heatmaps on the created subplots
    # Corrected the ax argument from `ax=axes=[0]` to `ax=axes[0]`
    sns.heatmap(correlation_matrix_pearson, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axes[0])
    axes[0].set_title('Pearson Correlation Heatmap')

    sns.heatmap(correlation_matrix_spearman, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axes[1])
    axes[1].set_title('Spearman Correlation Heatmap')

    plt.tight_layout()
    plt.show()
elif mode == '3':
    # MACHINE LEARNING
    # ----- MACHINE LEARNING TRAINING -----
    X = df_clean.drop(['SSPL', 'delta'], axis = 1)
    y = df_clean['SSPL']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate' : [0.05, 0.1],
        'max_depth' : [5, 7],
        'random_state' : [42]
    }

    model = XGBRegressor(random_state= 42)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', verbose=1)

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    test_predictions = best_model.predict(X_test)
    accuracy = r2_score(y_test, test_predictions)
    error = mean_squared_error(y_test, test_predictions, squared=False)

    print(f'\nModel Accuracy (R-squared): {accuracy *100:.2f}%')
    print(f'Model Error (MSE): {error:.4f}')

    # ----- MACHINE LEARNING INPUT -----
    print('\nENTER NEW AIRFOIL PARAMETERS')

    bounds = {
        'f': (df_clean['f'].min(), df_clean['f'].max()),
        'alpha': (df_clean['alpha'].min(), df_clean['alpha'].max()),
        'c': (df_clean['c'].min(), df_clean['c'].max()  ),
        'U_infinity': (df_clean['U_infinity'].min(), df_clean['U_infinity'].max()),
        'delta': (df_clean['delta'].min(), df_clean['delta'].max())
    }
    while True:
        print('\n-----------------------------------')
        print("Type 'exit' to end the program")
        
        questions = {
            'f': 'Frequency (Hz)',
            'alpha': 'Angle of Attack (deg)',
            'c': 'Chord Length (m)',
            'U_infinity': 'Velocity (m/s)',
        }
        
        new_parameter = {} 
        user_wants_to_exit = False
        
        for variable_name, prompt_text in questions.items():
            min_val, max_val = bounds[variable_name]
            full_prompt = f"{prompt_text} [{min_val:.2f} to {max_val:.2f}]: "
            while True:
                user_input = input(full_prompt)
                if user_input.lower() == 'exit':
                    print('Exiting program...')
                    user_wants_to_exit = True
                    break 
                try:
                    num_input = float(user_input)
                    
                    if num_input < min_val or num_input > max_val:
                        print(f"[!] Warning: {num_input} is out of bounds! The AI cannot extrapolate accurately. Please try again.")
                        continue
                    
                    new_parameter[variable_name] = [num_input]
                    break 
                except ValueError:
                    print("[!] Error: Please enter a valid number or type 'exit'.")
            if user_wants_to_exit:
                break
        if user_wants_to_exit:
            break
        if len(new_parameter) == 4:
            new_data = pd.DataFrame(new_parameter)
            prototype_prediction = best_model.predict(new_data)[0]
            print(f'\n>>> PREDICTED SSPL: {prototype_prediction:.4f} dB <<<')
else :
    print('\nType better dude')
    print('F# You!') # (FANTASTIC)
    exit()