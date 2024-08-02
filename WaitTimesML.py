# Import required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from datetime import datetime, timedelta
import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from datetime import datetime as dataframe
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Read .csv's fomr MIMIC-IV dataset
diagnosisdf = pd.read_csv(r"C:\Users\athar\OneDrive\Documents\UBC 4th Year\Thesis\mimic-iv-ed-2.2\ed\diagnosis.csv\diagnosis.csv")
edstaysdf = pd.read_csv(r"C:\Users\athar\OneDrive\Documents\UBC 4th Year\Thesis\mimic-iv-ed-2.2\ed\edstays.csv\edstays.csv")
triagedf = pd.read_csv(r"C:\Users\athar\OneDrive\Documents\UBC 4th Year\Thesis\mimic-iv-ed-2.2\ed\triage.csv\triage.csv")
medrecondf = pd.read_csv(r"C:\Users\athar\OneDrive\Documents\UBC 4th Year\Thesis\mimic-iv-ed-2.2\ed\medrecon.csv\medrecon.csv")
pyxisdf = pd.read_csv(r"C:\Users\athar\OneDrive\Documents\UBC 4th Year\Thesis\mimic-iv-ed-2.2\ed\pyxis.csv\pyxis.csv")
vitalsdf = pd.read_csv(r"C:\Users\athar\OneDrive\Documents\UBC 4th Year\Thesis\mimic-iv-ed-2.2\ed\vitalsign.csv\vitalsign.csv")

# CLEANUP DATAFRAMES
# rename columns to avoid duplicate column titles
medrecondf = medrecondf.rename(columns={'charttime': 'histcharttime'})
vitalsdf = vitalsdf.rename(columns={'charttime': 'vitcharttime'})

# convert columns with timestamps into datetime values and store columns as int
medrecondf['histcharttime'] = pd.to_datetime(medrecondf['histcharttime'], format='mixed')
edstaysdf['intime'] = pd.to_datetime(edstaysdf['intime'], format='mixed')
edstaysdf['outtime'] = pd.to_datetime(edstaysdf['outtime'], format='mixed')
pyxisdf['charttime'] = pd.to_datetime(pyxisdf['charttime'], format='mixed')
vitalsdf['vitcharttime'] = pd.to_datetime(vitalsdf['vitcharttime'], format='mixed')
triagedf['stay_id'] = triagedf['stay_id'].astype('int')
edstaysdf['gender'] = edstaysdf['gender'].replace({'M': 1, 'F': 0})

# Filter for UTIs: Pylonephritis & Unspecified UTIs
diagnosisdf = diagnosisdf.drop_duplicates(subset=['stay_id'], keep=False)  # keep line remove duplicates to filter only for UTIs
icd_uti_codes = ('N390', '5990', 'N10', 'N11', '59010', '59011')  # icd 9 and 10 included
icd_bti_codes = ('S06', '800', '804', '850', '854')
bti_conditions = '|'.join(icd_bti_codes)
uti_conditions = '|'.join(icd_uti_codes)
uti_df = diagnosisdf[diagnosisdf['icd_code'].str.startswith((icd_uti_codes))]  # Filter according to uti codes or bti codes, adjust as needed

# Create a set of stay_IDs that meet the "UTI" conditions and apply to medications administered
uti_set = set(uti_df['stay_id'])
pyxisdf = pyxisdf.loc[pyxisdf['stay_id'].isin(uti_set)]

# Function to simplify medication names using regex patterns
def simplify_med_name(name):
    # Normalize the medication name: remove parts that start with numbers
    simplified_name = re.sub(r'\b\d+[^ ]*', '', name)
    # Remove volume, brands and other details while keeping primary medication name and method of administration (tablet, vaccine, etc.)
    simplified_name = re.sub(r'\s*\d*\.?\d*\s*(mg|gm|ml)[^a-zA-Z]*', '', simplified_name, flags=re.IGNORECASE)
    simplified_name = re.sub(r'\s*\([^)]*\)', '', simplified_name)  # Remove anything in parentheses
    simplified_name = re.sub(r'\s*vial|bag', '', simplified_name, flags=re.IGNORECASE)  # Remove specific package types
    simplified_name = simplified_name.strip().lower()  # Convert to lower case and strip whitespace
    return simplified_name

# Apply the simplification function to the medication column and create a df with new med names and frequency
pyxisdf['simplified_medication'] = pyxisdf['name'].apply(simplify_med_name)
countsdf = pyxisdf['simplified_medication'].value_counts()
countsdf = countsdf.reset_index()
countsdf.columns = ['simplified_medication', 'count']
oneoffs = list(countsdf.loc[countsdf['count'] <= 3]['simplified_medication'])
oneoff_df = pyxisdf.loc[pyxisdf['simplified_medication'].isin(oneoffs)]
er_oneoff = edstaysdf.loc[edstaysdf['stay_id'].isin(oneoff_df['stay_id'])]
er_oneoff['staytime'] = er_oneoff['outtime'] - er_oneoff['intime']
edstayuti = edstaysdf.loc[edstaysdf['stay_id'].isin(uti_set)]
edstayuti['stay'] = edstayuti['outtime'] - edstayuti['intime']
meanstay = np.mean(edstayuti['stay'])
meanoneoff = np.mean(er_oneoff['staytime'])

# Set limit to only keep top 75% of medications
medcount_lim = sum(countsdf['count']) * 0.75
tally = 0
indexcounter = 0

# Filter out lower 25% of medications
for index, medrow in countsdf.iterrows():
    add = medrow['count']
    tally += add
    indexcounter += 1
    if tally >= medcount_lim:
        count_filter = countsdf.iloc[:indexcounter, :]
        break

simp_med_100 = count_filter['simplified_medication']

# Initialize a color array with 'blue' for all items
colors = ['blue'] * len(countsdf)

# Update the color array to 'red' for the bottom 25%
for idx in countsdf.index:
    if idx > indexcounter:
        colors[idx] = 'red'

# Plot the bar graph to show top relevant vs irrelevant medications
plt.bar(countsdf.index, countsdf['count'], color=colors)
plt.title('Bar Chart Representing Frequency of Medications Administered for BTIs')
red_patch = mpatches.Patch(color='red', label='Bottom 25%')
blue_patch = mpatches.Patch(color='blue', label='Top 75%')
plt.legend(handles=[red_patch, blue_patch])
plt.xlabel('Medication')
plt.ylabel('Log Scale Frequency of Medication')
plt.yscale('log')
plt.show()

# Remove duplicates and filter further to only include required stay_id's
pyxisdf_singular = pyxisdf.drop_duplicates(subset=['stay_id', 'simplified_medication']).reset_index()

pyxisdf_singular = pyxisdf_singular.loc[pyxisdf_singular['simplified_medication'].isin(simp_med_100)]

uti_set = pyxisdf_singular['stay_id']
medrecondf = medrecondf.loc[medrecondf['stay_id'].isin(uti_set)]
uti_df = uti_df.loc[uti_df['stay_id'].isin(uti_set)]

dataframe = pyxisdf_singular[['stay_id', 'simplified_medication']]

# Develop a function for creating a transposed dataframe
def create_transposed_df(df):
    # Pivot the table to get unique ids as rows and unique medications as columns
    # Fill missing values with 0, as those combinations do not exist in the input table
    transposed_df = pd.crosstab(df['stay_id'], df['simplified_medication'])
    
    # Convert the table to use 1s and 0s instead of counts (which are 1s and 0s in this case already)
    transposed_df = transposed_df.applymap(lambda x: 1 if x > 0 else 0)
    
    # Reset the index to make 'id' a column again and rename the columns
    transposed_df.reset_index(inplace=True)
    transposed_df.columns = ['stay_id'] + ['med' + col.upper() for col in transposed_df.columns[1:]]
    
    return transposed_df

# Develop a function for encoding the previously transposed dataframe
def encode_pad_transpose(df, stay_id_col, medication_col):
    """
    Encodes and pads the medication data of a DataFrame based on stay_id, and provides a dictionary of encoded values to medication names.

    Parameters:
    - df: DataFrame containing the data.
    - stay_id_col: Name of the column containing stay IDs.
    - medication_col: Name of the column containing medication names.

    Returns:
    - DataFrame with stay_id, and padded, encoded medication columns.
    - Dictionary of encoded values to medication names.
    """
    
    # Step 1: Encode 'medication_col'
    label_encoder = LabelEncoder()
    df = df.sort_values(by=[stay_id_col, medication_col])
    df['med_encoded'] = label_encoder.fit_transform(df[medication_col]) + 1  # +1 so 0 can be used for padding

    # Create a dictionary of encoded values to medication names
    value_to_med_name = {index + 1: label for index, label in enumerate(label_encoder.classes_)}

    # Step 2: Group by 'stay_id' and collect encoded values into lists
    grouped = df.groupby(stay_id_col)['med_encoded'].apply(list).reset_index(name='med_list')

    # Step 3: Find the maximum list length for padding
    max_len = grouped['med_list'].str.len().max()

    # Step 4: Pad lists and transpose into separate columns
    for i in range(max_len):
        grouped[f'med{i+1}'] = grouped['med_list'].apply(lambda x: x[i] if i < len(x) else 0)

    # Drop the 'med_list' column and ensure the DataFrame is in the desired format
    grouped.drop(columns=['med_list'], inplace=True)

    return grouped, value_to_med_name

# Apply the functions and merge dataframes respective to create a cohesive table to be fed to the model
admstrd_grouped = encode_pad_transpose(dataframe, 'stay_id', 'simplified_medication')[0]
admstrd_dict = encode_pad_transpose(dataframe, 'stay_id', 'simplified_medication')[1]
recon_grouped = encode_pad_transpose(medrecondf, 'stay_id', 'etccode')[0]
uti_grouped = encode_pad_transpose(uti_df, 'stay_id', 'icd_code')[0]
triage = triagedf[['stay_id','acuity', 'temperature', 'heartrate']]

binarymed = create_transposed_df(dataframe)
triage = triagedf[['stay_id','acuity', 'temperature', 'heartrate', 'resprate', 'sbp', 'dbp', 'o2sat']]
merge_df2 = pd.merge(uti_grouped, binarymed, on='stay_id', how='inner')
merge_df1 = pd.merge(edstaysdf[['stay_id','gender']], merge_df2, on='stay_id', how='inner')
merge_df = pd.merge(triage, merge_df1, on='stay_id', how='inner')
binary_df = merge_df.iloc[:, 1:]

merge_df2 = pd.merge(uti_grouped, admstrd_grouped, on='stay_id', how='inner')
merge_df1 = pd.merge(edstaysdf[['stay_id','gender']], merge_df2, on='stay_id', how='inner')
merge_df = pd.merge(triage, merge_df1, on='stay_id', how='inner')
merge_df = merge_df.iloc[:, 1:]
merge_df = merge_df.dropna()

# Plotting the respective variables to visualize the data (change variables as necessary)
plt.hist(merge_df['sbp'], bins=100, alpha=0.7, label='systolic blood pressure')  # Adjust the number of bins as necessary
plt.xlabel('Patient Sysstolic Blood Pressure')
plt.ylabel('Frequency')
plt.title('Histogram of BTI Patient Systolic Blood Pressure at Arrival')
plt.xlim(min(merge_df['sbp']), 250)
plt.legend()
plt.yscale('log')  # Set y-axis to logarithmic scale

plt.show()

# MODEL TRAINING AND TESTING
# Separate the features and the target variables
X = merge_df.iloc[:, :-10]
y = merge_df.iloc[:, -10:]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, random_state=42)

# Create the MultiOutputClassifier
multi_target_rf = MultiOutputClassifier(rf, n_jobs=-1)

# Train the model
multi_target_rf.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = multi_target_rf.predict(X_test)

# Collecting scores for each output
precisions = []
recalls = []
f1_scores = []

# Iterate through to calculate the scores respectively
for i in range(y_test.shape[1]):
    precisions.append(precision_score(y_test.iloc[:, i], y_pred[:, i], average='macro'))
    recalls.append(recall_score(y_test.iloc[:, i], y_pred[:, i], average='macro'))
    f1_scores.append(f1_score(y_test.iloc[:, i], y_pred[:, i], average='macro'))

print(precisions)
print(recalls)
print(f1_scores)

# Averaging the scores across all outputs
overall_precision = np.mean(precisions)
overall_recall = np.mean(recalls)
overall_f1 = np.mean(f1_scores)

print("Overall Precision: {:.4f}".format(overall_precision))
print("Overall Recall: {:.4f}".format(overall_recall))
print("Overall F1 Score: {:.4f}".format(overall_f1))

# Due to having multiple targets, we need to calculate accuracy for each one separately
accuracies = y_test.columns.map(lambda col: accuracy_score(y_test[col], y_pred[:,list(y_test.columns).index(col)]))

y_test_array = y_test.to_numpy()

# Develop a function to calculate accuracy by comparing the predicted array (pred_array) with the test data array (test_array)
def calculate_accuracy_and_frequency(pred_array, test_array):
    # Ensure both arrays have the same shape
    assert pred_array.shape == test_array.shape, "Arrays must have the same shape."

    # Overall accuracy
    overall_accuracy = np.mean(pred_array == test_array)
    print(f"Overall Accuracy: {overall_accuracy:.2%}")

    # Total number of elements in test_array
    total_elements = test_array.size - np.count_nonzero(test_array == 0)

    acc_list = []
    freq_list = []

    # Accuracy and frequency for each number from 0 to 24
    for number in range(1, np.max(test_array)):
        mask = test_array == number
        num_occurrences = np.sum(mask)
        if num_occurrences == 0:
            print(f"number {number} is not present")
        else:
            accuracy = np.mean(pred_array[mask] == test_array[mask])
            frequency = num_occurrences / total_elements
        acc_list.append(accuracy)
        freq_list.append(frequency)

    return acc_list, freq_list

# Apply accuracy function
acc_list1 = calculate_accuracy_and_frequency(y_pred, y_test_array)[0]
freq_list1 = calculate_accuracy_and_frequency(y_pred, y_test_array)[1]

# Plot bar graph showing the model accuracy
# Positions of the left bar-groups
barWidth = 0.3
r1 = np.arange(1, len(acc_list1)+1)
r2 = [x + barWidth for x in r1]

# Create blue bars
plt.bar(r1, acc_list1, color='blue', width=barWidth, edgecolor='grey', label='Accuracy')

# Create red bars (middle of group)
plt.bar(r2, freq_list1, color='red', width=barWidth, edgecolor='grey', label='Frequency')

# General layout
plt.xticks([r for r in range(1, 1 + len(acc_list1))])
plt.ylabel('Frequency/Accuracy')
plt.xlabel('Encoded Medication')
plt.title('Dual Bar Chart for BTIs Showing Model Class Accuracy vs Class Frequency in Dataset')
plt.legend()
plt.show()

# Separate method to calculate accuracy, since if a medicaiton is predicted in column 1 however is present in column 2, the prediction should still be treated as correct
ypred = pd.DataFrame(y_pred)
count = 0
count1 = 0
TP_count = 0
FP_count = 0
FN_count = 0
accuracy = []
Prec = []
Recall = []
F_1 = []
unique_case_pred = set()
unique_case_act = set()

# Iterate through and calculate true accuracy, positives, negatives, etc.
for i, row in ypred.iterrows():
    # Accuracy Calculation
    for x in range(0, 10):
        if row[x] != 0:
            if row[x] in list(y_test.iloc[i, :]):
                count += 1
                TP_count += 1
            else:
                FP_count += 1
        elif row[x] == y_test.iloc[i, x]:
            count += 1
            TP_count += 1
        else:
            FN_count += 1
    accuracy.append(count/10)
    prec = (TP_count/(TP_count + FP_count))
    if TP_count + FN_count != 0:
        recall = (TP_count/(TP_count + FN_count))
    else:
        recall = 0
    if prec + recall != 0:
        F1 = (2 * prec * recall)/(prec + recall)
    else: 
        F1 = 0
    Prec.append(prec)
    Recall.append(recall)
    F_1.append(F1)
    if (count/10) >= .9:
        count1 += 1
        processed_row = sorted([y for y in row if y != 0])
        unique_case_pred.add(tuple(processed_row))

    processed_row_act = sorted([z for z in row if z != 0])
    unique_case_act.add(tuple(processed_row_act))
    count = 0
    TP_count = 0
    FP_count = 0
    FN_count = 0

unique_pred_list = [list(row) for row in unique_case_pred]
unique_act_list = [list(row) for row in unique_case_act]

# Develop function to derive back to medication name
def replace_with_med_name(num_list, num_to_med_dict):
    replaced_list = []
    for sublist in num_list:
        med_names = [num_to_med_dict[num] for num in sublist if num in num_to_med_dict]
        replaced_list.append(med_names)
    return replaced_list

# Replace numbers with medication names
med_names_list = replace_with_med_name(unique_pred_list, admstrd_dict)

print(med_names_list)
print(count1/len(y_test))

print('random forests classifier model accuracy:', np.round(np.mean(accuracy)*100, 3), '%')
print('random forests classifier model prec:', np.round(np.mean(Prec)*100, 3), '%')
print('random forests classifier model recall:', np.round(np.mean(Recall)*100, 3), '%')
print('random forests classifier model f1:', np.round(np.mean(F_1)*100, 3), '%')

# Separate the features and the target variables
X = merge_df.iloc[:, :-10]  # All columns except the last seven
y = merge_df.iloc[:, -10:]  # Just the last seven columns

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)

# Create the MultiOutputClassifier wrapping the DecisionTreeClassifier
multi_target_dt = MultiOutputClassifier(dt, n_jobs=-1)

# Train the model
multi_target_dt.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = multi_target_dt.predict(X_test)

# Since we have multiple targets, calculate accuracy for each one
accuracies = {col: accuracy_score(y_test[col], y_pred[:, list(y_test.columns).index(col)]) for col in y_test.columns}

ypred = pd.DataFrame(y_pred)
count = 0  # to count overall cases
count1 = 0  # to count cases where prediction accuracy is high (in this case >= 90%)
TP_count = 0
FP_count = 0
FN_count = 0
accuracy = []
Prec = []
Recall = []
F_1 = []
unique_case_pred = set()
unique_case_act = set()

for i, row in ypred.iterrows():
    # Accuracy Calculation
    for x in range(0, 10):
        if row[x] != 0:
            if row[x] in list(y_test.iloc[i, :]):
                count += 1
                TP_count += 1
            else:
                FP_count += 1
        elif row[x] == y_test.iloc[i, x]:
            count += 1
            TP_count += 1
        else:
            FN_count += 1
    accuracy.append(count/(10))
    prec = (TP_count/(TP_count + FP_count))
    recall = (TP_count/(TP_count + FN_count))
    F1 = (2 * prec * recall)/(prec + recall)
    Prec.append(prec)
    Recall.append(recall)
    F_1.append(F1)
    # print('acc:', count/7, '\n')
    if (count/10) >= .9:
        count1 += 1
        processed_row = sorted([y for y in row if y != 0])
        unique_case_pred.add(tuple(processed_row))
    
    # if F1 >= 0.9:
    #     print(row, y_test.iloc[i, :])

    processed_row_act = sorted([z for z in row if z != 0])
    unique_case_act.add(tuple(processed_row_act))
    count = 0
    TP_count = 0
    FP_count = 0
    FN_count = 0

print('decision trees classifier model accuracy:', np.round(np.mean(accuracy)*100, 3), '%')
print('decision trees classifier model prec:', np.round(np.mean(Prec)*100, 3), '%')
print('decision trees classifier model recall:', np.round(np.mean(Recall)*100, 3), '%')
print('decision trees classifier model f1:', np.round(np.mean(F_1)*100, 3), '%')

# Optionally, print the accuracies for each target
for target, acc in accuracies.items():
    print(f"Accuracy for {target}: {acc}")

# EVENT LOG APPLICATION TO CREATE PROCESS FLOWS
# Create event log and decode the medications used
def create_eventlog(pred_meds_list):
    holder = 1
    df_list = []
    for meds in pred_meds_list:
        meds_len = len(meds)
        event_list = ['entry', 'medhist', 'vitals'] + meds + ['exit']

        time_list = []
        for num in range(0, meds_len + 4):
            time = datetime(2020, 1, 1, 0, 0, 0)
            time_int = time + timedelta(minutes=10*num)
            time_list.append(time_int)

        df = pd.DataFrame({
            'case:concept:name': [str(holder)] * len(event_list),
            'concept:name': event_list,
            'time:timestamp': time_list
        })

        holder += 1
        df_list.append(df)
    return df_list

# Store event log as a dataframe to be read later
df1 = create_eventlog(pred_meds_list=med_names_list)
master_df = pd.DataFrame()

for frames in df1:
    master_df = pd.concat([master_df, frames], ignore_index=True)

string_range = [str(i) for i in range(1, 15+1)]

df_1 = master_df.loc[master_df['case:concept:name'].isin(string_range)]

# Create process flows using pm4py library, change accordingly to compare different treatment pathways
log = pm4py.convert_to_event_log(df_1)
net, initial_marking, final_marking = alpha_miner.apply(log)
gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters={"node_font_size": 20, "edge_font_size": 20})
pn_visualizer.view(gviz)



