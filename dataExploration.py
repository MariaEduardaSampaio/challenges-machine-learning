import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def statistics(var):
    min_val = var.min()
    max_val = var.max()
    mean_val = var.mean()
    med_val = var.median()
    mod_val = var.mode()[0]
    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val, mean_val, med_val,
                                                                                             mod_val, max_val))


def show_density(var):
    plt.figure(figsize=(10, 4))
    var.plot.density()
    plt.title('Data Density')
    plt.axvline(x=var.mean(), color='cyan', linestyle='dashed', linewidth=2)
    plt.axvline(x=var.median(), color='red', linestyle='dashed', linewidth=2)
    plt.axvline(x=var.mode()[0], color='yellow', linestyle='dashed', linewidth=2)
    plt.show()


data = [50, 50, 47, 97, 49, 3, 53, 42, 26, 74, 82, 62, 37, 15, 70, 27, 36, 35, 48, 52, 63, 64]
grades = np.array(data)

study_hours = [10.0, 11.5, 9.0, 16.0, 9.25, 1.0, 11.5, 9.0, 8.5, 14.5, 15.5, 13.75, 9.0, 8.0, 15.5, 8.0, 9.0, 6.0, 10.0,
               12.0, 12.5, 12.0]

student_data = np.array([study_hours, grades])

avg_study = student_data[0].mean()
avg_grade = student_data[1].mean()

print('Average study hours: {:.2f}\nAverage grade: {:.2f}'.format(avg_study, avg_grade))

df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie',
                                     'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
                                     'Jakeem', 'Helena', 'Ismat', 'Anila', 'Skye', 'Daniel', 'Aisha'],
                            'StudyHours': student_data[0],
                            'Grade': student_data[1]})

# verifies how many null values exist in which column
print(df_students.isnull().sum())

'''
if there is a number missing, you can assume that this student studied
for an average amount of time and replace the missing value with the mean 
of that column, using the fillna() method

df_students.StudyHours = df_students.StudyHours.fillna(avg_study)

or you can only use data you know to be absolutely correct, so you can
drop rows or columns that contains null values by using the dropna() method

df_students = df_students.dropna(axis=0, how='any')
'''

passes = pd.Series(df_students.Grade >= 60)
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)

df_students = df_students.sort_values('Grade', ascending=False)
print(df_students)

'''
If you want to plot only one graph in a figure: 

plt.title('Student Grades')
plt.bar(x=df_students.Name, height=df_students.Grade)
plt.xticks(rotation=90)
plt.show()

'''

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Student Data')

ax[0].bar(x=df_students.Name, height=df_students.Grade, color='orange')
ax[0].set_title('Grades')

pass_counts = df_students.Pass.value_counts()
ax[1].pie(pass_counts, labels=pass_counts)
ax[1].set_title('Passing Grades')
ax[1].legend(pass_counts.keys().tolist())

fig.show()

var_grades = df_students.Grade
statistics(var_grades)

plt.figure(figsize=(10, 4))
plt.hist(var_grades)
plt.title('Data distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

plt.boxplot(var_grades)
plt.title('Data distribution')
plt.show()

for col_name in ['Grade', 'StudyHours']:
    col = df_students[col_name]
    rng = col.max() - col.min()
    var = col.var()
    std = col.std()
    print('\n{}:\n - Range: {:.2f}\n - Variance: {:.2f}\n - Std.Dev: {:.2f}'.format(col_name, rng, var, std))

print(df_students.describe())

# Rid any rows that contain outliers
df_sample = df_students[df_students.StudyHours > 1]

scaler = MinMaxScaler()
df_normalized = df_sample[['Name', 'Grade', 'StudyHours']].copy()

# Normalize the numeric columns
df_normalized[['Grade', 'StudyHours']] = scaler.fit_transform(df_normalized[['Grade', 'StudyHours']])

'''
a correlation measurement above 0 indicate that high values of one
 variable tend to coincide with high values of the other, while values below 0
 indicate a negative correlation, when high values of one variable tend to coincide
 with low values of the other 
'''
print(df_normalized.Grade.corr(df_normalized.StudyHours))