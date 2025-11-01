## Infotact_Project_01
#### Design and develop an intelligent task management system that leverages NLP and ML techniques to automatically classify, prioritize, and assign tasks to users based on their behavior, deadlines, and workloads.

## Step 1: Data Collection & Preprocessing
### **--Objective--**
The goal was to load the raw task dataset, clean it, enrich it with synthetic values (like deadlines and workloads), and prepare it for downstream machine learning tasks. This included:
<br>

:) Loading the dataset from Google Drive.
<br>

:) Removing duplicates and handling missing values.
<br>

:) Assigning synthetic deadlines and workloads.
<br>

:) Generating task priorities based on deadlines and workloads.
<br>

:) Automatically assigning users based on skill matching.
<br>

:) Preprocessing task descriptions using NLP techniques.
<br>

:) Saving the final cleaned dataset for future use.
<br>

### **--Key Steps--**
###  Data Loading
:) Shape of dataset → (20122, 8)
<br>

### Data Cleaning & Renaming
:) Removed duplicates.
<br>
:) Dropped rows with missing values in Task Description and Skill.
<br>
:) Shape of dataset now → (789, 8)

###  Dataset Exploration
:) Total unique task descriptions: 265
<br>
:) Total unique categories: 13
<br>
:) Total unique skills: 232
<br>

###  Skill Assignment to Users
:) A dictionary was created mapping 40 users to 3–4 random skills each
<br>

###  Deadline, Workload & Priority Assignment
:) **Deadline:** Random date within 60 days from today.
<br>

:) **Workload:** Random integer from 1 to 10.
<br>

:) **Priority Logic:**      
if days_left <= 7 or workload >= 8 → High  
elif days_left <= 20 → Medium  
else → Low
<br>

###  Task Assignment Based on Skills
:) Each task was assigned to a user who has the required skill. If no perfect match, a random user was assigned.
<br>

###  Text Preprocessing using NLP
Preprocessing on Task Description involved:
<br>
:) Lowercasing
<br>
:) Tokenizing
<br>
:) Removing stopwords & punctuation
<br>
:) Stemming
<br>

### Final Conclusion :
:) Cleaned Dataset Saved As: cleaned_dataset.csv
<br>
:) Download Triggered via: files.download('cleaned_dataset.csv')
<br>


## Step 2: Task Classification using NLP and ML
### **--Objective--**
To classify tasks into relevant categories using the task descriptions. Applied NLP preprocessing and trained two machine learning models – Naive Bayes and Support Vector Machine (SVM) – for multi-class classification.
<br>

### **--Key Steps--**
### Model Used :
:) Multinomial Naive Bayes
<br>
:) Linear Support Vector Classifier (SVM)
<br>

###  Dataset:
Used the cleaned dataset generated in step 1 which includes:
<br>
:) Processed_Description (preprocessed task text)
<br>
:) Category (target class)
<br>
:) Skill, Deadline, Priority, User Skills, and Assigned User columns
<br>

###  Results & Evaluation
**Naive Bayes:**
<br>
:) **Accuracy:** 94.30%
<br>
:) **Precision:** 94.20%
<br>
:) **Recall:** 94.30%
<br>

**Notable high scores in:**
<br>
:) Frontend, Documentation, Testing, Project Management – All achieved 95–100% accuracy
<br>
:) Slight drop in performance for rare categories like ui/ux design due to low support
<br>

**SVM:**
<br>
:) **Accuracy:** 95.57%
<br>
:) **Precision:** 95.90%
<br>
:) **Recall:** 95.57%
<br>

**Improved overall performance compared to Naive Bayes**
<br>
:) Consistently high performance across most classes
<br>
:) Handles imbalanced data better
<br>
:) Perfect scores in multiple categories including DevOps, Documentation, Project Management
<br>

### Final Conclusion :
Both models performed exceptionally well, but SVM slightly outperformed Naive Bayes in terms of accuracy and precision. It is more suitable for production-level task classification due to its better handling of class imbalance.


## Step 3: Priority Prediction & Recommended User Assignment
### **--Objective--**
In step 3, we focused on building a machine learning model to predict the priority of tasks based on multiple features and recommend the most suitable user for each task based on skills and workload. We experimented with Random Forest and XGBoost classifiers.
<br>

### **--Key Steps--**
### Data Preprocessing: :
:) Dropped rows with missing Priority values.
<br>

:) Encoded the Assigned User and Priority columns using LabelEncoder.
<br>

:) Calculated Days Left to the deadline 
<br>

:) Transformed User Skills into a numerical value using a simple heuristic: count of comma-separated skills.
<br>

###  Feature Set :
X = df[['User Skills', 'Workload', 'Assigned User Encoded', 'Days Left']]
<br>

y = df['Priority Encoded']
<br>

###  Train-Test Split :
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
<br>

###  **Random Forest Classifier :**
:) **Hyperparameter Tuning with GridSearchCV:**
<br>

param_grid = {
<br>
    'n_estimators': [50, 100],
<br>
    'max_depth': [None, 10, 20],
<br>
    'min_samples_split': [2, 5]
<br>
}
<br>

:) **Best Params:**
<br>
Best RF Params: {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100}
<br>

:) **Classification Report:**
![image](https://github.com/user-attachments/assets/f1b3a089-47ef-45a5-bfe3-8ed7ad94a144)
            
<br>

### **XGBoost Classifier:**
:) **Hyperparameter Tuning with GridSearchCV:**
<br>

xgb_param_grid = {
<br>
    'n_estimators': [50, 100],
<br>
    'max_depth': [3, 5, 10],
<br>
    'learning_rate': [0.01, 0.1, 0.2]
<br>
}
<br>

:) **Best Params:**
<br>
Best XGBoost Params: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50}
<br>

:) **Classification Report:**
![image](https://github.com/user-attachments/assets/932c98f8-8209-4b1c-a567-11bf9205f47a)

<br>

### **Recommended User Assignment:**
Used a simple logic to assign tasks to users with matching skills and lowest workload.
<br>



### **Final Output Sample:**
![image](https://github.com/user-attachments/assets/4b699157-d34a-4e5a-affd-d64d62c20ade)
<br>

## Step 4: Final Priority Prediction & Task Classification
### **--Objective--**
In step 4, we trained the final Random Forest model on the full dataset using the best hyperparameters found earlier, and classified tasks into types based on their skill requirements.
<br>

### **Approach:**
:) Trained the final Random Forest classifier with tuned parameters.
<br>
:) Created a simple function to categorize tasks into Task Types like Programming, Data Analysis, or General Task based on keywords in the User Skills.
<br>
:) Predicted task priorities and evaluated model performance.
<br>


###  Results:
**Random Forest Accuracy:** 1.00 (perfect accuracy on full dataset)
<br>
**Classification Report:**
<br>
![image](https://github.com/user-attachments/assets/5c11d3c6-1f16-4a83-9e52-3e45a0c365ac)
<br>


###  Visualizations:
:) Confusion matrix showing model predictions :
<br>
![image](https://github.com/user-attachments/assets/cbc95287-ed18-47a2-891b-b07e143693af)
<br>

:) Feature importance bar chart for key features
<br>
![image](https://github.com/user-attachments/assets/266462d0-f3a2-4333-8085-c1538a210a23)
<br>

:) Distribution plots for predicted priority levels and task types
<br>
![image](https://github.com/user-attachments/assets/c1b51f9a-75a2-4b18-a342-efefe2b3448f)
<br>
![image](https://github.com/user-attachments/assets/366e2b06-78d3-4532-91f9-cb5a63fa4775)
<br>


### Output:
The final summary table with predicted priorities and task types is saved as Final_Task_Summary.csv for further use.














