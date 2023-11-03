# SA4-AIML
GitHub for Samsung SA4 FS_AI4Media Evaluation
This repository is for the upload of scripts related to the evaluation for FS_AI4Media.

The repository has the following structure:

datasets: the datasets that are used in this evaluation framework will be hosted on external cloud storage and links will be provided in the README file of this folder
models: models that are proposed as reference may be hosted here using LFS or a link should be provided to an external location, where the model is accessible. Candidate models should be hosted externally.
scripts: the evaluation scripts are structured into seperate folders, one for each scenario. They include scripts to run inference and evaluate different metrics that pertain to that specific scenario. Generic scripts that apply to multiple scenarios may be directly under the scripts folder.
docker: the docker folder will host the Dockerfile that is used to generate a container image, which can be used to instantiate the evaluation environment as a docker container.
