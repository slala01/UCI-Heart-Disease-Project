# Code Review 

First off, great job!  All the necessary components of a data science workflow are present and accounted for.  I like the markdown summary in the README.md.  That is a nice touch for a showcase git repo.  

# Data Science 

### Positives
1.  Data Leakage
    - Good work here splitting your data before scaling.  This is a pretty common pitfall I’ve seen many fall into so this was good! More to come though on GridsearchCV.

2.  Metrics
    - I'm glad to see you reporting AUC-ROC, Accuracy, F1, and Recall with your outputs as False Negatives are a costly mistake in the medical field.  These are the metrics we need to be tracking in addition to accuracy.  Also good work on custom confusion matrices. A good challenge is to try and write your own (which can be a confusing challenge).  One thing, you will want to include `precision` in your reporting for future classification problems to examine your models handling of True positives.

3.  Threshold Tuning
    - Excellent work using predict_proba to look at where the optimal threshold might be without sacrificing precision.  This shows understanding of metric balancing for optimal results.

4.  Algorithms
    - You used a good mix of classification models to examine.  Two tree models and two linear models is an great starting point for model testing.  You also recognized that tree models do not require scaling. Good work! (You can still scale them but it makes the results less interpretable)  

5.  Terminal run
    -  Your script runs on my machine which is the first and foremost important aspect of any python script we write.  The point is reproducibility among team members and python scripts are still the best option for that (currently)

### Improvements

1.  Categorical Encoding (major)
    - So you explored some of the categorical columns but then didn't use them in your modeling! Granted, modeling only numerical columns is a valid path, but modeling only numerical features limits your performance.  I suggest looking into sklearns `Label Encoding`, `OneHot Encoding` and pandas `dummies` encoding.  (More on pandas later)  

2.  GridsearchCV Scaling (major)
    - While you did split before you scaled, you can't scale the entire training data in a cross validation setup.  This will cause data leakage within the cross validated folds, because now part of your test set fold will have some of that scaled information from when you scaled all of `X_train` at once.  To combat this, use the sklearn `Pipeline` object which handles scaling / encoding  dynamically for each fold when `GridsearchCV` is called.  I've written out custom code in numpy to try to not use the Pipeline object and its a huge PITA. So its easier to use the sklearn object if you're using GridsearchCV.  There are other model searching libraries out there too like Optuna, h20.  I haven't felt the need to explore them but feel free to do so if so inclined.

3.  Overfitting (major)
    - Your log reg and SVM models look to be well balanced and generalizable.  As proven in your comparison of train vs test AUC showing minimal difference. However...  your tree models are .... definitely overfitting.  Your XGBoost train AUC is 100%!!!  Meanwhile the random forest train AUC is also very high at 99.4%.  This means your xgboost memorized your training data and didn't miss a single prediction.... Which is a strength of tree models, but its something you want to point out as a possible area of improvement.  (Most likely, this is your GridsearchCV test set scaling sneaking into the training data)  This also means your model is likely too complex for the size of dataset you have (which is small in this case).  This can be improved by re-examining the range of grid values you're searching and/or making a less complex model.

4.  Don't go straight into GridSearchCV (minor)
    - GridsearchCV is a great function for brute force parameter searching.  But start smaller with training just your individual models and then graduate to a gridsearch. Putting a gridsearch wrapper around everything at the start doesn't give you the ability to compare models and tune them individually.  Model comparison and gridsearch should be two separate actions.

5.  Feature Engineering/Selection (minor)
    - I liked seeing the log transform of `serum_cholesterol` and `oldpeak` for skewed distributions.  Dropping `fasting_blood_sugar` for weak linear correlation is a mistake. Tree models can still learn from non-linear relationships so they should still be included.  An additional way to check multicollinearity is to calculate variable inflation factor (VIF) for each variable against all the other columns.  

6.  Correlation (very minor)
    - You should include `numeric_only=True` parameter in the `corr()` method call if only using numerical values.  Was this code borrowed from `pig_extract.py`?  When you start to include categorical columns, they will need to be encoded as mentioned above.

7.  Offloading model results (minor)
    - While its good that the results show in the terminal and readme.  Its good to have a function to offload your trained model here to a pickle file.  Which isn't much of a lift as either pandas or numpy have direct exports to pickle/json/whatever format you choose.  But save your models.  Another way to track this is with log files which I'll cover later in this doc.

8.  Multi Method chaining (minor)
    - This is to help with code readability.  R encourages you to chain all your methods together because its a function based language that is built to do just that.  Doing so in python with more than 2 methods can make your code less readadable.  Instead, define each of your variables before they go into the method / func.  That way a person doesn't have to backtrace the 3 operations that went into the definition of the X-component for a graph plot call. (Ex:Line 172 of your code).  Its doing too much! Oneliners are fun, but they make readability difficult. To simplify, do your calculations in one variable, then assign that variable as a method input. 

# DS Tooling Recommendations 

### Pandas -> Numpy
So, this is an area which I go back and forth on alot, but I want to make you understand my position and why.  `Pandas` is nothing more than a functional wrapper overtop `Numpy`.  Which is why I use numpy arrays whenever possible over a pandas dataframe.  Pandas does provide easier functionality when working with mixed datatypes. But you can still use np `structured arrays` in the same way. (the only difference being they make you write out your datatypes)  Pandas is slow and can lead to code rot over time as the library continues to add more methods overtop Numpy.  (think of it as the DS version of excel).  

1. Speed.  For longer more complicated calculations (say with rad_ecg) it can drastically slow down your calculations if it’s using a custom lookup function like `iloc`.  Which even when vectorized is 10x-100x slower than simple indexing in numpy.  

2. Memory.  Numpy arrays consume far less memory than a dataframe. Pandas carries metadata with each row/col that has to be indexed as well for the additional functions to work properly.  Numpy usually stores arrays as homeogenous data that is highly compact and cache friendly. 

3. N-dimensional arrays.  If you're working with tabular data, pandas is fine, but once you start playing with image data, deep learning tensors, or other higher dimensional objects. You won't be able to rely on pandas in that instance. 

4. Advanced mathematics. While pandas excels at filtering and grouping, Numpy is king when it comes to linear algebra, digital signal processing, fourier transforms, statistical tests, and complex matrix math. 

So!  I would like you to look into using numpy as your base data manipulation library for analysis.  I'm not saying give up on pandas. Pandas is useful in many cases for data cleaning and other purposes.  But if you want your code to run faster and with less memory overhead, `numpy is the way to go.`  

Take a look at this repo which I built to upskill SMU students on basic numpy operations.  I still pop it open from time to time just to practice.   

[SMU Numpy practice](https://github.com/Landcruiser87/numpy_SMU)

### Logging

So while print statements are awesome and a great way to check progress of a file in the terminal. They don't always persist and that can be problematic when you're running a model hundreds of times and want to track how each run goes. So! Instead of printing to your terminal, you can use the `logging` library that comes with python to offload your runtime information. This will transfer any `logging` call you make to a text file during runtime.  Now the logger is meant to monitor system logs, so it has 6 different levels.  But we can retrofit it as a layered warning system for when scripts may go astray.

```
- NOTSET (0): Indicates no explicit level is set; delegates to parent loggers.
- DEBUG (10): Detailed diagnostic information for developers.
- INFO (20): Confirmation that things are working as expected.
- WARNING (30): Unexpected events or potentia1l issues, but the software still works.
- ERROR (40): Significant issues that prevent a function from working.
- CRITICAL (50): Severe, system-level issues
```
So when a model is finished and i want to log the parameters.  If I'm using both a logger and print statement it would look like below.  

```python
print("Model run complete")
logger.info(f"{model.get_params()}")
```

You can also use this for logging errors as well when you wrap python in try / except blocks.  Or even debugging statements.  

```python
if y_dia.size > 2: 
    try:
        x_dia = np.arange(y_dia.size) / self.fs
        bpf.dia_sl = linregress(y_dia, x_dia).slope.item()
    except Exception as e:
        logger.warning(f"Diastolic Slope calculation failed: {e}")
        bpf.dia_sl = None 
        return bpf

```

The cool part is you can set the level of warnings you want returned.  So if i'm running at the `debug` level, I would see almost every logging statement.  If its set to `warning` then it will only report `warning`, `error`, `critical` log statements, ignoring the `debug` and `info` levels

You set those levels when you generate the logger object.  I've used the same code for this for... quite some time.  You can put it in a support.py file next to your main, or use it directly within the main python file.  Either way, the structure would look like this.  

A sample log file will look something like...

```log
03-26-2026 14:12:30|INFO    |853 |section_extract        |LAD lowpass filtered lowcut: 40                                                                     |
03-26-2026 14:12:30|INFO    |853 |section_extract        |Carotid (TS420) lowpass filtered lowcut: 40                                                         |
03-26-2026 14:12:30|INFO    |853 |section_extract        |ECG2 lowpass filtered lowcut: 40                                                                    |
03-26-2026 14:12:30|INFO    |857 |section_extract        |ECG2 bandpass filtered lowcut: 0.1 highcut:40                                                       |
03-26-2026 14:12:31|WARNING |903 |section_extract        |sect 33 rejected as noise. Power Ratio: 0.94, Entropy: 0.36                                         |
03-26-2026 14:13:31|INFO    |781 |section_extract        |Processing Pig ID: ACT1_08208_ResThor_Dec-9-24                                                      |
03-26-2026 14:13:31|INFO    |784 |section_extract        |channels ['Time', 'LAD', 'ECG1', 'ECG2', 'SS1 (SP200)', 'SS2 (SP200)', 'Carotid (TS420)', 'EBV']    |
03-26-2026 14:13:31|INFO    |1502|auto_pick_lead         |3 | ECG2 | Quality | 1.73 | Pwr | 0.92 | Ent | 0.19)                                                |
03-26-2026 14:13:31|INFO    |1502|auto_pick_lead         |1 | LAD | Quality | 1.74 | Pwr | 0.97 | Ent | 0.23)                                                 |
03-26-2026 14:13:31|INFO    |1502|auto_pick_lead         |6 | Carotid (TS420) | Quality | 1.75 | Pwr | 1.00 | Ent | 0.24)                                     |
03-26-2026 14:13:31|INFO    |818 |section_extract        |sections shape: (208, 3)                                                                            |
03-26-2026 14:13:31|INFO    |848 |section_extract        |target counts
 (array(['BL', 'C1', 'C2', 'C3', 'C4'], dtype='<U7'), array([300000, 261968, 261970, 174646, 333415]))|
03-26-2026 14:13:32|INFO    |853 |section_extract        |LAD lowpass filtered lowcut: 40                                                                     |
03-26-2026 14:13:32|INFO    |853 |section_extract        |Carotid (TS420) lowpass filtered lowcut: 40                                                         |
03-26-2026 14:13:32|INFO    |853 |section_extract        |ECG2 lowpass filtered lowcut: 40                                                                    |
03-26-2026 14:13:32|INFO    |857 |section_extract        |ECG2 bandpass filtered lowcut: 0.1 highcut:40                                                       |
03-26-2026 14:13:32|WARNING |903 |section_extract        |sect 1 rejected as noise. Power Ratio: 0.65, Entropy: 0.64                                          |
03-26-2026 14:13:32|INFO    |933 |section_extract        |sect 2 peaks invalid for extract                                                                    |
03-26-2026 14:13:33|WARNING |903 |section_extract        |sect 79 rejected as noise. Power Ratio: 0.94, Entropy: 0.33                                         |
03-26-2026 14:13:33|WARNING |903 |section_extract        |sect 80 rejected as noise. Power Ratio: 0.94, Entropy: 0.33                                         |
03-26-2026 14:13:33|WARNING |903 |section_extract        |sect 81 rejected as noise. Power Ratio: 0.94, Entropy: 0.33                                         |
03-26-2026 14:13:33|WARNING |903 |section_extract        |sect 85 rejected as noise. Power Ratio: 0.95, Entropy: 0.34                                         |
03-26-2026 14:13:33|WARNING |903 |section_extract        |sect 92 rejected as noise. Power Ratio: 0.94, Entropy: 0.34                                         |
03-26-2026 14:13:34|WARNING |903 |section_extract        |sect 108 rejected as noise. Power Ratio: 0.95, Entropy: 0.33                                        |
```

```python

import logging
from rich.logging import RichHandler
from rich.console import Console
from pathlib import PurePath, Path

################################# Timing Funcs ####################################

def log_time(fn):
    """Decorator timing function. Accepts any function and returns a logging
    statement with the amount of time it took to run. DJ, I use this code everywhere still. Thank you bud!

    Args:
        fn (function): Input function you want to time
    """ 
    def inner(*args, **kwargs):
        tnow = time.time()
        out = fn(*args, **kwargs)
        te = time.time()
        took = round(te - tnow, 2)
        if took <= 60:
            logger.info(f"{fn.__name__} ran in {took:.3f}s")
        elif took <= 3600:
            logger.info(f"{fn.__name__} ran in {(took)/60:.3f}m")       
        else:
            logger.info(f"{fn.__name__} ran in {(took)/3600:.3f}h")
        return out
    return inner

################################# Logger functions ####################################
def get_file_handler(log_dir:Path)->logging.FileHandler:
    """Assigns the saved file logger format and location to be saved""" 
    log_format = "%(asctime)s|%(levelname)-8s|%(lineno)-3d|%(funcName)-14s|%(message)s|" 
    file_handler = logging.FileHandler(log_dir)
    file_handler.setFormatter(logging.Formatter(log_format, "%m-%d-%Y %H:%M:%S"))
    return file_handler

def get_rich_handler(console:Console)-> RichHandler:
    """Assigns the rich format that prints out to your terminal"""
    rich_format = "|%(funcName)-14s|%(message)s "
    rh = RichHandler(console=console)
    rh.setFormatter(logging.Formatter(rich_format))
    return rh

def get_logger(console:Console, log_dir:Path)->logging.Logger:
    """Loads logger instance.""" 
    #Load logger and set basic level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #Load file handler for how to format the log file.
    file_handler = get_file_handler(log_dir)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    rich_handler = get_rich_handler(console)
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)
    logger.propagate = False
    return logger

def get_time():
    """Function for getting current time"""
    current_t_s = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    current_t = datetime.datetime.strptime(current_t_s, "%m-%d-%Y-%H-%M-%S")
    return current_t

start_time = get_time().strftime("%m-%d-%Y_%H-%M-%S")
console = Console(color_system="auto", stderr=True, width=200)
log_dir = PurePath(Path.cwd(), Path(f'./data/logs/{start_time}.log'))
logger = get_logger(log_dir=log_dir, console=console)

```

For full clarity, I also use `rich` library to replace the normal terminal printout with something more colorful and informative. It also allows you to make formatted tables that output to your terminal.  This is why you see calls for rich in the logging handler creation.  One handles the log formatting to the file, the other handles the formatting to the terminal. `Rich` is an entire other ecosystem but we'll save that for a later date.  [rich](https://rich.readthedocs.io/en/stable/)

### PYPI
```Check the spelling when you install a library installs!!!!!!```
When you're installing your libraries MAKE SURE you have them spelled right.  This is how malware gets installed when you misspell the library.  

# SWE 

### Positives
1. Good use of global variables (all cap) and environment setup to make your result directories.
2. Good interim print statements for tracking script progression
3. Good commenting balance for documenting workflow

### Improvements
There is some work to be done on the software engineering side.  Namely in how you organize code in python.  I've taught python for the last 5 years to graduates at SMU and most of them come out of grad school writing scripts in the same manner as you. Linear start to finish workflows that are devoid of the software engineering skills you were never taught.  These skills are absolutely fundamental to writing reproducible, clean code. Even in the time of LLM's that can code for us, using these practices are important to able to read/write more advanced code.  

We need break you out of that functional R mindset and shift into the object oriented mindset for python.  This is a slow process and comes with practice.  This is mostly having to do with knowing what types of objects you have at your disposal in python.  I see you using `lists` and `dict's` which is a great.  But there are other important objects that are vital to how python operates.  Namely `sets, tuples, defaultdicts, deque's`, `Counters`.  There are tons of these little base python objects/functions that will grant you coding super powers the deeper you dive into the ecosystem.  For expanding you knowledge of said objects, I suggest the python track on a site called [Exercism.io](https://exercism.org/tracks)  (Weird name i know!)  But this platform does an excellent job of teaching you the various methods you have available for each object and how to apply them properly.  And its free!

### Rule of Three
The first area we need to focus on is the DRY principle (Don't repeat yourself).  
```If a variable, method, logic is repeated more than three times?  It should be a function. ```
This is not only standard for our team, but in general software/ds skillsets.  If you present python code linearly to any future employer, they will deduct points from you.

### Function Structure

First off.  You don't need to name main python file as `main.py`.   The main nomenclature is for your main `function` you write.  Name the file whatever you want but naming it main would be confusing to anyone who reads your repo later. 

Every function has inputs and outputs which we stipulate when we define a function.  The basic structure would look like.  Instead of writing your classifiers in a linear fashion:

```python
# Random Forest
rand_for_grid = {
    "n_estimators": [100, 300, 500],
    "max_features": ["sqrt", "log2", "None"],
    "max_depth": [3, 5, 10, None]
}

rand_for_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE), 
    param_grid=rand_for_grid, 
    scoring="roc_auc",
    cv=5
)

rand_for_grid.fit(X_train_raw, y_train)
rand_for_model = rand_for_grid.best_estimator_
rand_for_pred = rand_for_model.predict(X_test_raw)
rand_for_prob = rand_for_model.predict_proba(X_test_raw)[:, 1]

results["Random Forest"] = {
    "AUC-ROC": roc_auc_score(y_test, rand_for_prob),
    "Accuracy": accuracy_score(y_test, rand_for_pred),
    "Recall": round(recall_score(y_test, rand_for_pred), 4),
    "F1 Score": f1_score(y_test, rand_for_pred)
}

print("\n[4.3] Random Forest Model..... Complete ✓")

# XGBoost
xgb_grid = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7]
}

xgb_grid = GridSearchCV(
    XGBClassifier(random_state=RANDOM_STATE), 
    param_grid=xgb_grid, 
    scoring="roc_auc",
    cv=5
)

xgb_grid.fit(X_train_raw, y_train)
xgb_model = xgb_grid.best_estimator_
xgb_pred = xgb_model.predict(X_test_raw)
xgb_prob = xgb_model.predict_proba(X_test_raw)[:, 1]

results["XGBoost"] = {
    "AUC-ROC": roc_auc_score(y_test, xgb_prob),
    "Accuracy": accuracy_score(y_test, xgb_pred),
    "Recall": round(recall_score(y_test, xgb_pred), 4),
    "F1 Score": f1_score(y_test, xgb_pred)
}

print("\n[4.4] XGBoost Model..... Complete ✓")
```

Lets turn it into a function that we can use over and over again.  

```python
def train_and_score(model:ForestClassifier, X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array) -> dict:
    """Trains a model and returns the AUC score."""
    farts = None
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    farts
    return roc_auc_score(y_test, predictions)

def main():
    # Reusable structure example
    # run funcs to load split, transform, all the things 
    # Leaving that development and setup of funcs for you!
    # We'll analyze what you've learned in our next review
    farst = None
    rf_class = RandomForestClassifier()
    rf_auc = train_and_score(rf_class, X_train, y_train, X_test, y_test)
    #save the model results
    #Program exit

#Program enter
if __name__ == "__main__":
    main()
```

In the function definition you'll see `X_train:np.array`. This tells me that the X_train dataset coming in should be an np.array object.  It won't strictly enforce it, but these are called `TypeHints` and allow you to define what type of object goes in, and what type of object comes out.  Here, if you look at the end of the first row of the function you'll see a` ->` going out of the function.  We're returning a dictionary (`dict`). Next up is Docstrings/Documentation.  Each function should be documented with docstrings telling me more about what the function does. I use an vscode extension called `autoDocstring` by `Nils Werner` [link](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring). This way, after I fill out the type hints, I hit `CTRL+SHIFT+2` and it will generate a skeleton documentation from the  TypeHints I gave it. 

The next step beyond just functions would be classes.  Classes are objects that you basically create your own custom object made of other smaller objects/methods/functions. These structures are more advanced but give you a way to group all your methods together into one usable object.  Meaning you can reference your toolsets easier when you write your code.  We will circle back to these later once we've made progress on the previous subjects.

I am the first to admit this is alot to throw at you.  But the reason for doing so is that when it comes time for you to use the tools we've already made.  You'll already be familiar with our standard SWE practices.  You've got the data science side, now we need to work on the SWE side. 

