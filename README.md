# EE8223-Final-Project

## About the Project
This project is the implementation code for the paper:
Bahrami, S., Chen, Y. C., and Wong, V. W. S., “Deep reinforcement learning for demand response in distribution networks,” IEEE Transactions on Smart Grid, vol. 12, no. 2, pp. 1496–1506, Mar. 2021.

Please read the project report for more details about this paper and implementation tricks.

## Built With
Jupyter Notebook

## File Descriptions on GitHub
0. dayprof : function used by ML_Inputs
1. InputData_HouseholdStates : synthetic data created as input for Python script, Main Actor Critic Demand Response
2. Main Actor Critic Demand Response : main code in .py and .ipynb which ever you would like to refer to
3. ML_Inputs : script to read in Raw_Data_Calculation to output synthetic data InputData_HouseholdStates
4. Raw_Data_Calculation : spreadsheet used to build raw data based on research included and documented here
5. tmp/actor_critic : weights saved for actor-critic model

## Installations
Before running this code, make sure the following libraries are installed:

* pandas
  ```sh
  conda install pandas  
  ```

* numpy
  ```sh
  conda install numpy
  ```

* random
  ```sh
  pip install random
  ```
  
* tensorflow
  ```sh
  pip install --upgrade tensorflow
  ```
  
* tensorflow_probability
  ```sh
  pip install --upgrade tensorflow-probability
  ```
  
* math
  ```sh
  pip install python-math
  ```
  
* matplotlib
  ```sh
  pip install matplotlib
  ```

## Steps to Run Project
0. Download and save all files
1. Run ML_Inputs.m in MATLAB which will call on dayprof.m and Raw_Data_Calculation.xls
2. ML_Inputs.m will output InputData_HouseholdStates.xls
3. Run Main Actor Critic Demand Response.py which calls on InputData_HouseholdStates.xls to get all results shown in report * make sure all installations above are complete*


## Acknowledgements
I would like to acknowledge the following examples that helped me learn, structure and implement this code:
* https://keras.io/examples/rl/actor_critic_cartpole/
* https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/actor_critic/tensorflow2

