# If you are getting errors or not getting the output in PART 1 then try PART 2

# -------------PART 1-------------
> pip install -r requirements.txt
## Run the main.py file with following as arguments 
## uai file 
## task id (1-fod learn, 2-pod learn, 3-mixture model)
## train_file_path
## test_file_path
## k value if task id = 3

for e.g.
> python main.py hw5-data/dataset1/1.uai 2 hw5-data/dataset1/train-p-1.txt hw5-data/dataset1/test.txt
> python main.py hw5-data/dataset1/1.uai 3 hw5-data/dataset1/train-f-1.txt hw5-data/dataset1/test.txt 4





# -------------PART 2-------------
# Steps to run the code... commands are tested in linux.. you can apply alternative commands for windows/MacOS
## Step 1 creating a virtual environment to run the code so that it does not conflicts with other instaled packages on the machine
> python3 -m venv my_env
## Step 2 if the above gives error then make sure your python version is 3.6 or above and install the venv package. If no error move to Step 3
	### for linux and MacOS
	> python3 -m pip install --user virtualenv
	### for windows
	> py -m pip install --user virtualenv

## Step 3 activate the environment
> source my_env/bin/activate


## Step 2 use requirements.txt file to install required packages
> pip install -r requirements.txt

## once done use the part 1 commands to run the output


### once done with grading of the code you can deactivate the environment and delete it
> deactivate
> rm -r my_env
