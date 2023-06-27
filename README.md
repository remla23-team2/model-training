# Model-training
Contains the ML training pipeline. 
This repository is dedicated to storing all training-related files for the Machine Learning (ML) part of the course. 

## Instructions

### Clone the Repository
Create a local copy of this repository by cloning it 
```
git clone https://github.com/remla23-team2/model-training.git
```
  
#### Install Requirements
Run the following commands from your terminal in the application folder:
```
pip install -r requirements.txt
```

#### Run the main
In your terminal, run the following commands:
```
python main.py
```

## Docker
Open the terminal (in the application folder) and run the following commands to create a Docker image:
```shell script
docker build -t ghcr.io/remla23-team2/model-training:VERSION .
```

### DVC DAG

```
 +----------+  
 | get_data |  
 +----------+  
       *       
       *       
       *       
+------------+ 
| preprocess | 
+------------+ 
       *       
       *       
       *       
  +-------+    
  | train |    
  +-------+
       *
       *
       *
 +----------+
 | evaluate |
 +----------+
 ```
### DVC DAG Outputs

```
+----------------------------+ 
| ..\output\getdata\data.tsv |
+----------------------------+
               *
               *
               *
   +----------------------+
   | ..\output\preprocess |
   +----------------------+
               *
               *
               *
      +-----------------+
      | ..\output\train |
      +-----------------+
               *
               *
               *
    +--------------------+
    | ..\output\evaluate |
    +--------------------+
 ```

### Test Coverage Report

https://github.com/remla23-team2/model-training/suites/13904181358/artifacts/773493594

[![Coverage Report](https://github.com/remla23-team2/model-training/actions/artifacts/automated-testing/coverage-report/htmlcov/index.html)](https://github.com/remla23-team2/model-training/actions/artifacts/automated-testing/coverage-report/htmlcov/index.html)

