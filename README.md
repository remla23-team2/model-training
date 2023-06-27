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

### Code Coverage Report

![badge](https://gist.githubusercontent.com/izagorac/74faf8906ea0f22889d78cfd9c88171e/raw/10bfd03382e7b8274252fc99a96792b4c808790d/pytest-results.json) 

![badge](https://gist.githubusercontent.com/izagorac/9f559ce5704ca16aca7db02b79efe22f/raw/deafb39cf5ec0ab17c69e21d339e1af36332e46b/code-coverage.json)
