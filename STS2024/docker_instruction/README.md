# 📋 Docker Instruction

In the Docker container you build, please ensure that the `predict.sh` script exists in the `/opt/algorithm` directory.
We will call the `predict.sh` script from the outside to execute the corresponding segmentation model.
You can read one test sample to be predicted from the `/inputs` directory,
and save the test prediction to the `/outputs` directory.
We are providing a tutorial for a sequential execution Docker container here, which is unoptimized,
where all cases are executed one after another for prediction.

First, make sure that `Docker` and `NVIDIA Container Toolkit` are installed on your computing platform as they are essential for the algorithm packaging.
The former ensures that you can perform the packaging, while the latter enables GPU utilization within Docker.
Be sure to confirm that your system has been properly installed and configured.

Next, make sure to modify the `requirements.txt` file to include the necessary dependencies for your code project.
This ensures that all the required libraries for the prediction process are included so that the prediction code can be executed correctly and produce the desired results.

Then,
implement your inference testing process in the `run_inference.py` file.
After that, execute the `build.sh` script to troubleshoot any errors.
Building with `build.sh` might take a little while, so thanks for your patience!


Finally, proceed to execute the `export.sh` script to export the `sts24_algorithm_docker.tar.gz` file
that can be sent to our [STS-challenge Email](https://sts-challenge.github.io/miccai2024/index.html).
Each team has three opportunities to submit a Docker container, but we can only retain one successfully executed Docker.
Therefore, please ensure that your Docker contains the final version of your algorithm.
If we encounter any issues during the execution on the test set,
we will email you the relevant error information to ensure the correct execution of your Docker.
Also,
you can verify your exported Docker container using the [Execution Code](https://github.com/STS-challenge/STS/tree/main/STS2024/evaluation/resource_evaluation) we provide.

